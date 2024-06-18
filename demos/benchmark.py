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
from skimage import color
from kornia.color import rgb_to_lab
from PIL import Image

import torch
from torch.functional import F
import PIL
from kornia.color import rgb_to_lab

from scipy import stats

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
from demos.custom import PhotometricFitting

uv_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fitting = PhotometricFitting(device=device)

# Initialize face detection and landmark estimation
face_detect = detector.SFDDetector(device, cfg.rect_model_path)
face_landmark = FAN_landmark.FANLandmarks(device, cfg.landmark_model_path, cfg.face_detect_type)

# Load and prepare the segmentation network
seg_net = BiSeNet(n_classes=cfg.seg_class).to(device)
seg_net.load_state_dict(torch.load(cfg.face_seg_model))
seg_net.eval()

def process_image(image_path, output_filename, save_folder, device_name):
    """Processes a given image and performs photometric fitting."""
    # Prepare filenames for output
    save_name = f"{output_filename}.obj"
    save_video_name = f"{output_filename}.avi"
    video_path = os.path.join(save_folder, save_video_name)

    # Create a video writer object
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 16, (cfg.image_size, cfg.image_size * 8))

    # Initialize the fitting process

    img = cv2.imread(image_path)

    # Run the fitting process
    fitting.run(img, seg_net, face_detect, face_landmark, cfg.rect_thresh, save_name, video_writer, save_folder)

def load_image(path):
    """Loads an image and converts it to L*a*b color space."""
    return Image.open(path).convert('RGB')

def apply_mask(image, mask):
    """Applies a mask to the given image."""
    image.putalpha(mask)

def calculate_ita(image, mask):
    # Convert RGB to LAB
    lab_image = color.rgb2lab(image)
    L, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]
    # Apply mask: set values to NaN where mask is zero
    L[mask == 0] = np.nan
    b[mask == 0] = np.nan
    # Calculate ITA
    ita = np.arctan((L - 50) / b) * (180 / np.pi)
    return ita

def classify_skin_type(ita):
    skin_types = np.empty(ita.shape, dtype='object')
    # Classify each pixel based on ITA score
    skin_types[ita > 55] = 'Type I (Very light)'
    skin_types[(ita > 41) & (ita <= 55)] = 'Type II (Light)'
    skin_types[(ita > 28) & (ita <= 41)] = 'Type III (Intermediate)'
    skin_types[(ita > 10) & (ita <= 28)] = 'Type IV (Dark)'
    skin_types[(ita > -30) & (ita <= 10)] = 'Type V (Brown)'
    skin_types[ita <= -30] = 'Type VI (Black)'
    return skin_types

def log_ita_to_csv(csv_path, image_name, ita_gt, ita_gen, ita_error, skin_type, predicted_skin_type):
    """Logs the ITA values, error, and skin type for an image to a CSV file."""
    fieldnames = ['image_name', 'ita_gt', 'ita_gen', 'ita_error', 'skin_type', 'predicted_skin_type']
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
            'skin_type': skin_type,
            'predicted_skin_type': predicted_skin_type
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

def log_and_print(message, file):
    print(message)  # This will output to stdout
    file.write(message + '\n')  # This will write to the log file

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
    output_path = os.path.join(save_folder, 'xyz_output_files')
    csv_path = os.path.join(save_folder, '_____km_processed_images_log.csv')

    # Initialize the tqdm progress bar
    progress_bar = tqdm(total=batch_size, desc="Processing Images")

    # Load the mask for ITA calculation
    mask = Image.open(mask_path).convert('L')

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

            with open(log_file_path, 'w') as log_file:
                # Redirect stdout to both the log file and terminal
                original_stdout = sys.stdout  # Save a reference to the original standard output

                # Process the current image
                process_image(file, output_filename, output_path, device_name)

                # After processing, use the custom function to log and print the success message
                log_and_print(f"Successfully processed {file}, log saved to {log_file_path}", log_file)

                sys.stdout = original_stdout  # Restore stdout to original

            gt = Image.open(ground_truth_image_path).convert('RGB')
            gt.putalpha(mask)

            predicted = Image.open(generated_image_path).convert('RGB')
            predicted.putalpha(mask)

            mask_np = np.array(mask)

            # Apply mask and convert PIL Images to NumPy arrays
            gt.putalpha(mask)
            predicted.putalpha(mask)
            gt_np = np.array(gt)[:, :, :3]  # Drop alpha channel
            predicted_np = np.array(predicted)[:, :, :3]  # Drop alpha channel

            # Calculate ITA for both images, applying the mask
            ita_gt = calculate_ita(gt_np, mask_np)
            ita_predicted = calculate_ita(predicted_np, mask_np)

            # Compute mean difference in ITA where neither is NaN
            valid_mask = ~np.isnan(ita_gt) & ~np.isnan(ita_predicted)

            # Determine the skin type based on the true ITA value

            skin_types = classify_skin_type(ita_gt)
            predicted_skin_types = classify_skin_type(ita_predicted)

            mode = stats.mode(skin_types[~np.isnan(ita_gt)])
            skin_type = mode.mode[0]

            mode = stats.mode(predicted_skin_types[~np.isnan(ita_predicted)])
            predicted_skin_type = mode.mode[0]

            ita_gt = np.nanmean(ita_gt[valid_mask])
            ita_predicted = np.nanmean(ita_predicted[valid_mask])

            # mean_ita_difference = np.abs(np.abs(ita_gt) - np.abs(ita_predicted))
            mean_ita_difference = np.abs(ita_gt - ita_predicted)

            print(f"ITA Error for {output_filename}: {mean_ita_difference}")

            # Log the ITA calculation results and skin type to the CSV
            log_ita_to_csv(csv_path, output_filename, ita_gt, ita_predicted, mean_ita_difference, skin_type, predicted_skin_type)

            # Update logging and progress bar
            print(f"Processed image {processed_images + 1}/{batch_size}: {file}")
            progress_bar.update(1)

            # Increment the processed images counter
            processed_images += 1

        if processed_images >= batch_size:
            break

    progress_bar.close()  # Ensure the progress bar is closed properly