import os, sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import distutils.version
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.append('.')
from models.FLAME import FLAME, TextureModel, FLAMETex
from models.face_seg_model import BiSeNet
from models.RENI import RENI
from utils.renderer import Renderer
from utils import util
from utils.lr import CustomExponentialLR, CustomGroupLR
from utils.config import cfg
from facial_alignment.detection import sfd_detector as detector
from facial_alignment.detection import FAN_landmark

from reni_plus_plus.model_components.illumination_samplers import EquirectangularSamplerConfig
from reni_plus_plus.field_components.field_heads import RENIFieldHeadNames

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

class PhotometricFitting(object):
    def __init__(self, device='cuda'):
        self.config = cfg
        self.device = device
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config, TextureModel.BALANCED_ALBEDO_TRUST).to(self.device)
        self.reni = RENI(self.config).to(self.device)

        self._setup_renderer()
        self._setup_tensorboard()

    def _setup_renderer(self):
        directions = self.reni.get_light_directions()
        self.render = Renderer(cfg.image_size, obj_filename=cfg.mesh_file, light_directions=directions).to(self.device)

    def _setup_tensorboard(self):
        self.writer = SummaryWriter('runs/photometric_fitting_experiment_test_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    def optimize(self, images, landmarks, image_masks, video_writer):
        num_faces = images.shape[0]
        bz = num_faces  # batch size now represents the number of faces
        shape = nn.Parameter(torch.zeros(bz, cfg.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, cfg.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, cfg.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, cfg.pose_params).float().to(self.device))
        cam = torch.zeros(bz, cfg.camera_params)
        cam[:, 0] = 5.  # initial camera parameters
        cam = nn.Parameter(cam.float().to(self.device))

        # scale = nn.Parameter(torch.zeros(bz).float().to(self.device))
        # latent_codes = nn.Parameter(torch.zeros(bz, cfg.reni_latent_dim_size, 3).float().to(self.device))
        scale = nn.Parameter(torch.zeros(1).float().to(self.device))
        latent_codes = nn.Parameter(torch.zeros(1, cfg.reni_latent_dim_size, 3).float().to(self.device))

        # Learning rate configurations as before
        # albedo_lr, default_lr, latent_lr = cfg.e_lr, cfg.e_lr * 1.5, cfg.e_lr * 0.125
        albedo_lr, default_lr, latent_lr = cfg.e_lr, cfg.e_lr, cfg.e_lr * 0.5
        e_opt = torch.optim.Adam([
            {'params': [shape, exp, cam], 'lr': default_lr, 'weight_decay': cfg.e_wd, 'initial_lr': default_lr},
            {'params': [pose, tex], 'lr': albedo_lr, 'weight_decay': cfg.e_wd, 'initial_lr': albedo_lr},
            {'params': [scale, latent_codes], 'lr': latent_lr, 'weight_decay': cfg.e_wd, 'initial_lr': latent_lr}
        ])

        # Custom learning rate scheduler
        s_opt = CustomGroupLR(e_opt, [
            (700, [1.0, 0.0, 0.0]),
            # (300, [0.0, 1.0, 0.0]),
            (1500, [1.0, 1.0, 1.0]),
        ])

        for k in range(2000):
            e_opt.zero_grad()

            # Loss computation per face
            total_loss = 0.0
            all_landmarks2d, all_landmarks3d, all_vertices, all_trans_vertices = [], [], [], []
            all_face_renders, all_albedo_renders, all_albedos = [], [], []
            skip = (k + 1) % 50 != 0  # We skip detailed computation except every 50 iterations
            logging = (k + 1) % 10 == 0  # Log every 10 iterations
            non_rigid_mode = (k + 1) > 600

            for face_idx in range(num_faces):
                vertices, landmarks2d, landmarks3d = self.flame(
                    shape_params=shape[face_idx:face_idx+1],
                    expression_params=exp[face_idx:face_idx+1],
                    pose_params=pose[face_idx:face_idx+1]
                )
                trans_vertices = util.batch_orth_proj(vertices, cam[face_idx:face_idx+1])
                trans_vertices[..., 1:] = -trans_vertices[..., 1:]
                landmarks2d = util.batch_orth_proj(landmarks2d, cam[face_idx:face_idx+1])
                landmarks2d[..., 1:] = -landmarks2d[..., 1:]
                landmarks3d = util.batch_orth_proj(landmarks3d, cam[face_idx:face_idx+1])
                landmarks3d[..., 1:] = -landmarks3d[..., 1:]

                losses = {
                    'landmark': util.l2_distance(landmarks2d[:, :, :2], landmarks[face_idx][:, :2]),
                    'shape_reg': (shape[face_idx].pow(2).sum() / 2) * cfg.w_shape_reg,
                    'expression_reg': (exp[face_idx].pow(2).sum() / 2) * cfg.w_expr_reg,
                    'pose_reg': (pose[face_idx].pow(2).sum() / 2) * cfg.w_pose_reg,
                    'tex_reg': (tex[face_idx].pow(2).sum() / 2) * cfg.w_tex_reg,
                    'photometric_texture': torch.tensor(0.0),  # Initialize to zero
                    'exponential': torch.tensor(0.0)  # Initialize to zero
                }

                if non_rigid_mode:
                    predicted_illumination = self.reni(rotation=None, latent_codes=latent_codes, scale=scale)
                    # albedos = self.flametex(tex[face_idx]) / 255.
                    albedos = self.flametex(tex[face_idx:face_idx+1]) / 255.
                    ops = self.render(vertices, trans_vertices, albedos, illumination=predicted_illumination)
                    # predicted_images = ops['images']
                    losses['photometric_texture'] = F.smooth_l1_loss(image_masks[face_idx] * ops['images'], image_masks[face_idx] * images[face_idx]) * cfg.w_pho
                    losses['exponential'] = latent_codes.pow(2).mean() * 1e-4 * 10

                if not skip:
                    all_landmarks2d.append(landmarks2d)
                    all_landmarks3d.append(landmarks3d)
                    all_vertices.append(vertices)
                    all_trans_vertices.append(trans_vertices)

                    if non_rigid_mode:
                        all_face_renders.append(ops['images'])
                        all_albedo_renders.append(ops['albedo_images'])
                        all_albedos.append(albedos)

                face_loss = sum(losses.values())
                total_loss += face_loss

            total_loss.backward()
            e_opt.step()
            s_opt.step()


            if logging:
                self.writer.add_scalar('Total Loss', total_loss.item(), k)
                for key, loss in losses.items():
                    self.writer.add_scalar(f'Loss/{key}', loss.item(), k)

                print(f'Iter: {k+1}, Loss: {total_loss.item()}')

            if not skip:
                grids = {}
                visind = torch.arange(num_faces)  # Visualize all faces

                grids['images'] = torchvision.utils.make_grid(images[visind], nrow=num_faces, padding=0).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks[visind]), nrow=num_faces, padding=0)
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], torch.cat(all_landmarks2d, dim=0)), nrow=num_faces, padding=0)
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], torch.cat(all_landmarks3d, dim=0)), nrow=num_faces, padding=0)

                shape_images = self.render.render_shape(torch.cat(all_vertices, dim=0), torch.cat(all_trans_vertices, dim=0), images)
                grids['shape'] = torchvision.utils.make_grid(
                    F.interpolate(shape_images[visind], [224, 224]), nrow=num_faces, padding=0).detach().float().cpu()

                # Log the grids to TensorBoard
                self.writer.add_image('Images/Original', grids['images'], k)
                self.writer.add_image('Images/Landmarks_GT', grids['landmarks_gt'], k)
                self.writer.add_image('Images/Landmarks_2D', grids['landmarks2d'], k)
                self.writer.add_image('Images/Landmarks_3D', grids['landmarks3d'], k)
                self.writer.add_image('Images/Shape', grids['shape'], k)


                if non_rigid_mode:
                    albedos = torchvision.utils.make_grid(torch.cat(all_albedos, dim=0), nrow=num_faces).detach().cpu()
                    albedo_renders = torchvision.utils.make_grid(torch.cat(all_albedo_renders, dim=0), nrow=num_faces).detach().cpu()
                    renders = torchvision.utils.make_grid(torch.cat(all_face_renders, dim=0), nrow=num_faces).detach().cpu()
                    illumination_map = self.reni.visualize_illumination(predicted_illumination).permute(2, 0, 1).detach().cpu()

                    self.writer.add_image('Images/Albedo', albedos, k)
                    self.writer.add_image('Images/Albedo_Render', albedo_renders, k)
                    self.writer.add_image('Images/Render', renders, k)
                    self.writer.add_image('Images/Illumination', illumination_map, k)


                # Calculate frame dimensions for vertical stacking
                frame_width = grids['images'].shape[2]
                frame_height = sum(grid.shape[1] for grid in grids.values())
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                current_y = 0
                for key in ['images', 'landmarks_gt', 'landmarks2d', 'landmarks3d', 'shape']:
                    grid = grids[key].numpy().transpose((1, 2, 0))
                    grid = (grid * 255).astype(np.uint8)
                    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                    frame[current_y:current_y + grid.shape[0], :, :] = grid
                    current_y += grid.shape[0]

                video_writer.write(frame)


        return {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'latent_codes': latent_codes.detach().cpu().numpy(),
            'scale': scale.detach().cpu().numpy(),
        }

    def run(self, img, net, rect_detect, landmark_detect, rect_thresh, save_name, savefolder):
        # Detect all faces in the image
        bboxes = rect_detect.extract(img, rect_thresh)
        if not bboxes:
            print("No faces detected.")
            return

        # Initialize containers for multiple faces
        images = []
        landmarks = []
        image_masks = []

        # Process each detected face
        for bbox in bboxes:
            crop_image, new_bbox = util.crop_img(img, bbox, cfg.cropped_size)

            # Extract landmarks for the cropped face image
            resize_img, landmark = landmark_detect.extract([crop_image, [new_bbox]])
            landmark = landmark[0]
            landmark[:, 0] = landmark[:, 0] / float(resize_img.shape[1]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(resize_img.shape[0]) * 2 - 1
            landmarks.append(torch.from_numpy(landmark)[None, :, :].double().to(self.device))

            # Preprocess the image
            image = cv2.resize(crop_image, (cfg.cropped_size, cfg.cropped_size)).astype(np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            images.append(torch.from_numpy(image[None, :, :, :]).double().to(self.device))

            # Generate face segmentation masks
            image_mask = util.face_seg(crop_image, net, cfg.cropped_size)
            image_masks.append(torch.from_numpy(image_mask).double().to(self.device))

        # Convert lists to tensors
        images = torch.cat(images, dim=0)
        images = F.interpolate(images, [cfg.image_size, cfg.image_size])
        landmarks = torch.cat(landmarks, dim=0)
        image_masks = torch.cat(image_masks, dim=0)
        image_masks = F.interpolate(image_masks, [cfg.image_size, cfg.image_size])

        # Video writer setup for saving the optimization process
        util.check_mkdir(savefolder)
        save_video_name = save_name + '.avi'
        frame_size = (cfg.image_size * 3, cfg.image_size * 5)
        video_writer = cv2.VideoWriter(
            os.path.join(savefolder, save_video_name),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            16,  # Frame rate
            frame_size  # Width and height
        )

        if not video_writer.isOpened():
            print("Error: Could not open VideoWriter")
            exit()

        # Debug lines to visualize the images
        # Convert tensor to numpy array and rearrange dimensions for plotting
        # images_np = images.cpu().numpy()
        # images_np = images_np.transpose(0, 2, 3, 1)  # NCHW to NHWC
        # images_np = (images_np * 255).astype(np.uint8)  # Convert to uint8

        # landmarks_np = landmarks.cpu().numpy()

        # fig, axs = plt.subplots(1, len(images_np), figsize=(15, 5))
        # for i in range(len(images_np)):
        #     axs[i].imshow(images_np[i])
        #     axs[i].scatter(
        #         (landmarks_np[i, :, 0] + 1) / 2.0 * cfg.image_size,
        #         (landmarks_np[i, :, 1] + 1) / 2.0 * cfg.image_size,
        #         s=10, c='red', marker='o'
        #     )
        #     axs[i].axis('off')
        # plt.show()

        # # Optional: Visualize masks if needed
        # image_masks_np = image_masks.cpu().numpy()
        # image_masks_np = image_masks_np.transpose(0, 2, 3, 1)  # NCHW to NHWC

        # fig, axs = plt.subplots(1, len(image_masks_np), figsize=(15, 5))
        # for i in range(len(image_masks_np)):
        #     axs[i].imshow(image_masks_np[i], cmap='gray')
        #     axs[i].axis('off')
        # plt.show()

        params = self.optimize(images, landmarks, image_masks, video_writer)

        save_file = os.path.join(savefolder, save_name)

        # save single_params to a file
        # np.save(os.path.join(savefolder, save_name), single_params)
        np.savez(save_file, **params)

        for idx in range(len(bboxes)):
            shape = torch.from_numpy(params['shape'][idx:idx+1]).float().to(self.device)
            exp = torch.from_numpy(params['exp'][idx:idx+1]).float().to(self.device)
            pose = torch.from_numpy(params['pose'][idx:idx+1]).float().to(self.device)
            cam = torch.from_numpy(params['cam'][idx:idx+1]).float().to(self.device)
            tex = torch.from_numpy(params['tex'][idx:idx+1]).float().to(self.device)

            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]

            albedos = self.flametex(tex) / 255.

            filename = f'{save_file}_{idx}.obj'

            img = images[idx].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(savefolder, f"{save_file}_img_{idx}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            self.render.save_obj(
                filename=filename,
                vertices=trans_vertices[0],
                textures=albedos[0]
            )

        illumination = self.reni(rotation=None, latent_codes=torch.from_numpy(params['latent_codes']).float().to(self.device), scale=torch.from_numpy(params['scale']).float().to(self.device))
        illumination_map = F.interpolate(
            self.reni.visualize_illumination(illumination).permute(2, 0, 1).unsqueeze(0),
            size=(cfg.image_size, cfg.image_size*2),
            mode='bilinear',
            align_corners=False
        ).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(os.path.join(savefolder, f"{save_file}_map.png"), cv2.cvtColor(illumination_map, cv2.COLOR_RGB2BGR) * 255)

        # Release the VideoWriter
        video_writer.release()

if __name__ == '__main__':
    # image_path = str(sys.argv[1])
    # device_name = str(sys.argv[2])
    image_path = 'benchmarks/FAIR_benchmark/validation_set/full_image/ag_face_triplegangers_3_300_000161.png'
    image_path = 'benchmarks/FAIR_benchmark/validation_set/full_image/ag_face_triplegangers_3_300_000170.png'
    device_name = 'cuda'

    save_name = os.path.split(image_path)[1].split(".")[0]
    fitting = PhotometricFitting(device=device_name)
    img = cv2.imread(image_path)

    face_detect = detector.SFDDetector(device_name, cfg.rect_model_path)
    face_landmark = FAN_landmark.FANLandmarks(device_name, cfg.landmark_model_path, cfg.face_detect_type)

    seg_net = BiSeNet(n_classes=cfg.seg_class).cuda()
    seg_net.load_state_dict(torch.load(cfg.face_seg_model))
    seg_net.eval()

    fitting.run(img, seg_net, face_detect, face_landmark, cfg.rect_thresh, save_name,
                cfg.save_folder)
