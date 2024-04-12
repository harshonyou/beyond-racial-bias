import os
import sys
import cv2
import glob
import torch
import numpy as np
import argparse
import csv
from tqdm import tqdm
from contextlib import redirect_stdout

import torch
from torch.functional import F
import PIL
from kornia.color import rgb_to_lab

# Ensure the local directory is included in the Python path
sys.path.append('.')

# Import utilities and configurations
from utils import util
from utils.config import cfg

# Import model and detection components
from models.face_seg_model import BiSeNet
from facial_alignment.detection import sfd_detector as detector
from facial_alignment.detection import FAN_landmark

# Import fitting module
from demos.wj_fitting import PhotometricFitting

uv_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(image_path, output_filename, save_folder, device_name):
    """Processes a given image and performs photometric fitting."""
    # Prepare filenames for output
    save_name = f"{output_filename}.obj"
    save_video_name = f"{output_filename}.avi"
    video_path = os.path.join(save_folder, save_video_name)

    # Create a video writer object
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 16, (cfg.image_size, cfg.image_size * 7))

    # Initialize the fitting process
    fitting = PhotometricFitting(device=device_name)
    img = cv2.imread(image_path)

    # Initialize face detection and landmark estimation
    face_detect = detector.SFDDetector(device_name, cfg.rect_model_path)
    face_landmark = FAN_landmark.FANLandmarks(device_name, cfg.landmark_model_path, cfg.face_detect_type)

    # Load and prepare the segmentation network
    seg_net = BiSeNet(n_classes=cfg.seg_class).to(device_name)
    seg_net.load_state_dict(torch.load(cfg.face_seg_model))
    seg_net.eval()

    # Run the fitting process
    fitting.run(img, seg_net, face_detect, face_landmark, cfg.rect_thresh, save_name, video_writer, save_folder)

def load_image(path):
    """Loads an image and converts it to L*a*b color space."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2Lab)

def load_image2(path):
    image = PIL.Image.open(path).convert('RGB')
    image = np.asarray(image) / 255.
    image = torch.from_numpy(image[None, :, :, :]).permute(0,3,1,2).to(device)
    image = F.interpolate(image, (uv_size, uv_size))
    return image

def apply_mask(image, mask):
    """Applies a mask to the given image."""
    return cv2.bitwise_and(image, image, mask=mask)

def calculate_ita(lab_image):
    """Calculates the Individual Typology Angle (ITA) of a given image."""
    L, _, b = cv2.split(lab_image)
    ita = (np.arctan((L - 50) / b) * 180 / np.pi)[b != 0]  # Avoid division by zero
    return np.mean(ita)

def calculate_ita2(img, mask):
    img = rgb_to_lab(img)

    ITA = (img[:,0,:,:] - 50) / (img[:,2,:,:] + 1e-8)
    ITA = torch.atan(ITA) * 180 / torch.pi
    ITA = ITA[mask]

    return torch.mean(ITA)

def get_skin_type(ita_value):
    """Determines the skin type based on the ITA value."""
    if ita_value > 55:
        return 'Very light (I)'
    elif ita_value > 41:
        return 'Light (II)'
    elif ita_value > 28:
        return 'Intermediate (III)'
    elif ita_value > 10:
        return 'Tan (IV)'
    elif ita_value > -30:
        return 'Brown (V)'
    else:
        return 'Dark (VI)'

def log_ita_to_csv(csv_path, image_name, ita_gt, ita_gen, ita_error, skin_type):
    """Logs the ITA values, error, and skin type for an image to a CSV file."""
    fieldnames = ['image_name', 'ita_gt', 'ita_gen', 'ita_error', 'skin_type']
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Only write header if file does not exist
        writer.writerow({
            'image_name': image_name,
            'ita_gt': ita_gt,
            'ita_gen': ita_gen,
            'ita_error': ita_error,
            'skin_type': skin_type
        })

def check_if_image_processed(csv_path, image_name):
    """Checks if an image has already been processed and logged in the CSV."""
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['image_name'] == image_name:
                return True
    return False

def parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process images for photometric fitting and ITA calculation.")

    # Define arguments with defaults as per the previous hardcoded values
    parser.add_argument('--benchmark_root', type=str, default="benchmarks/FAIR_benchmark/validation_set/crops/", help="Path to the benchmark root directory.")
    parser.add_argument('--save_folder', type=str, default=os.path.join(cfg.root_dir, 'benchmarks_results'), help="Path to save processed images and results.")
    parser.add_argument('--device_name', type=str, default="cuda", help="Device name (e.g., 'cuda' or 'cpu').")
    parser.add_argument('--mask_path', type=str, default="benchmarks/FAIR_benchmark/validation_set/skin_for_ita_mask_cheeks.png", help="Path to the mask image for ITA calculation.")
    parser.add_argument('--batch_size', type=int, default=1, help="Number of images to process in this batch.")

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Use arguments to set variables
    benchmark_root = args.benchmark_root
    save_folder = args.save_folder
    device_name = args.device_name
    mask_path = args.mask_path
    batch_size = args.batch_size
    output_path = os.path.join(save_folder, 'output_files')
    csv_path = os.path.join(save_folder, '__processed_images_log.csv')

    # Initialize the tqdm progress bar
    progress_bar = tqdm(total=batch_size, desc="Processing Images")

    # Load the mask for ITA calculation
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    fair_mask = cv2.resize(cv2.imread(mask_path), (uv_size, uv_size)).astype(np.float32) / 255.
    fair_mask = torch.from_numpy(fair_mask[None, :, :, :]).permute(0,3,1,2).to(device)
    fair_mask = fair_mask[:,0,:,:] == 1.0

    # Ensure the save directory exists
    util.check_mkdir(output_path)

    # Initialize a counter for processed images
    processed_images = 0

    # Process each image in the benchmark directory
    for subdir, dirs, files in os.walk(benchmark_root):
        for file in glob.glob(os.path.join(subdir, '*.png')):
            if processed_images >= batch_size:
                progress_bar.close()
                print(f"Batch limit of {batch_size} images reached. Exiting.")
                break

            # Prepare paths and filenames
            output_filename = f"{os.path.split(os.path.dirname(file))[-1]}_{os.path.splitext(os.path.basename(file))[0]}"
            generated_image_path = os.path.join(output_path, f"{output_filename}.png")
            ground_truth_image_path = file.replace("crops", "crop-albedos")

            # Check if the image has already been processed
            if check_if_image_processed(csv_path, output_filename):
                print(f"Skipping already processed image: {output_filename}")
                continue

            print(f"Processing {file} as {output_filename}")

            # Determine the path for the log file for this image
            log_file_path = os.path.join(output_path, f"{output_filename}.log")

            # # Redirect stdout to the log file for this image
            with open(log_file_path, 'w') as log_file, redirect_stdout(log_file):
                # Process the current image
                process_image(file, output_filename, output_path, device_name)

                # After processing, manually print a success message to the original stdout
                sys.stdout = sys.__stdout__
                print(f"Successfully processed {file}, log saved to {log_file_path}")

            # Load and mask the ground truth and generated images
            # ground_truth_image = load_image(ground_truth_image_path)
            # generated_image = load_image(generated_image_path)
            # masked_gt = apply_mask(ground_truth_image, mask)
            # masked_gen = apply_mask(generated_image, mask)

            ground_truth_image = load_image2(ground_truth_image_path)
            generated_image = load_image2(generated_image_path)

            # Calculate and print ITA error
            # ita_gt = calculate_ita(masked_gt)
            # ita_gen = calculate_ita(masked_gen)
            # ita_error = abs(ita_gt - ita_gen)
            # print(f"ITA Error for {output_filename}: {ita_error}")

            ita_gt = calculate_ita2(ground_truth_image, fair_mask)
            ita_gen = calculate_ita2(generated_image, fair_mask)
            ita_error = abs(ita_gt - ita_gen)
            print(f"ITA Error for {output_filename}: {ita_error}")

            # Determine the skin type based on the true ITA value
            skin_type = get_skin_type(ita_gt)

            # Log the ITA calculation results and skin type to the CSV
            log_ita_to_csv(csv_path, output_filename, ita_gt.item(), ita_gen.item(), ita_error.item(), skin_type)

            # Update logging and progress bar
            print(f"Processed image {processed_images + 1}/{batch_size}: {file}")
            progress_bar.update(1)

            # Increment the processed images counter
            processed_images += 1

        if processed_images >= batch_size:
            break

    progress_bar.close()  # Ensure the progress bar is closed properly