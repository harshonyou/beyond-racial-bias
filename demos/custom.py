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
        # self.batch_size = cfg.batch_size
        # self.image_size = cfg.image_size
        # self.cropped_size = cfg.cropped_size
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
        self.writer = SummaryWriter('runs/photometric_fitting_experiment_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    def optimize(self, images, landmarks, image_masks, video_writer):
        bz = images.shape[0]
        shape = nn.Parameter(torch.zeros(bz, cfg.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, cfg.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, cfg.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, cfg.pose_params).float().to(self.device))
        cam = torch.zeros(bz, cfg.camera_params)
        cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))

        # latent_codes = nn.Parameter(torch.zeros(bz, cfg.reni_latent_dim_size, 3).float().to(self.device))
        scale = nn.Parameter(torch.zeros(bz).float().to(self.device))

        latent_codes = nn.Parameter(torch.zeros(bz, cfg.reni_latent_dim_size, 3).float().to(self.device))

        # mask = latent_codes > 0.75
        # latent_codes.data[mask] = 1
        # mask = latent_codes < 0.25
        # latent_codes.data[mask] = 0

        # e_opt = torch.optim.Adam(
        #     [shape, exp, pose, cam, tex, latent_codes, scale],
        #     lr=cfg.e_lr,
        #     weight_decay=cfg.e_wd
        # )
        # Configuration for learning rates
        # albedo_lr = cfg.e_lr * 0.25  # Assuming you want to increase the learning rate for albedo by 10x
        albedo_lr = cfg.e_lr * 1 #* 0.25 # 0.125  # 18 I
        default_lr = cfg.e_lr * 1.5  # Original learning rate for other parameters
        latent_lr = cfg.e_lr * 0.125 # 0.025 #0.01  # Half the original learning rate for latent codes

        e_opt = torch.optim.Adam([
            {'params': [shape, exp, cam], 'lr': default_lr, 'weight_decay': cfg.e_wd, 'initial_lr': default_lr},
            {'params': [pose, tex], 'lr': albedo_lr, 'weight_decay': cfg.e_wd, 'initial_lr': albedo_lr},  # Higher learning rate for albedo
            {'params': [scale, latent_codes], 'lr': latent_lr, 'weight_decay': cfg.e_wd, 'initial_lr': latent_lr}  # Lower learning rate for latent codes
        ])

        # s_opt = torch.optim.lr_scheduler.ExponentialLR(e_opt, gamma=0.95)
        # Define custom decay rates for each group
        # decay_rates = [0.99, 0.97, 0.95]  # 6 VI, 30ish I
        # intervals = [100, 250, 50]

        # 26 I
        # decay_rates = [0.99, 0.90, 0.95]  # For example, slower decay for tex, faster for latent_codes
        # intervals = [100, 250, 50]

        decay_rates = [1, 1, 1]  # For example, slower decay for tex, faster for latent_codes
        intervals = [100, 100, 100]

        rigid_mode = True

        # Example of phase settings
        # phase_settings = [
        #     (500, [1.0,     0.0,    0.0]),  # Phase 1: High LR for albedo, low for illumination
        #     (500, [0.25,    1.0,    0.1]),  # Phase 2: Low LR for albedo, high for illumination
        #     (250, [0.25,    0.1,    1.0]),  # Phase 3: Repeat Phase 1 settings
        #     (250, [0.25,    1.0,    0.1]),  # Phase 4: Repeat Phase 2 settings
        #     (500, [0.125,   0.5,    0.5])  # Final Phase: High LR for both
        # ]
        phase_settings = [
            (500, [1.0,     0.0,    0.0]),  # Phase 1: High LR for albedo, low for illumination
            (250, [0.0,     1.0,    0.0]),  # Phase 2: Low LR for albedo, high for illumination
            (1500, [1.0,     1.0,    1.0]),  # Phase 2: Low LR for albedo, high for illumination
        ]

        # phase_settings = [
        #     (1000, [1.0,     1.0,    1.0]),  # Phase 1: High LR for albedo, low for illumination
        # ]

        # Create the custom scheduler
        # s_opt = CustomExponentialLR(e_opt, decay_rates)
        # s_opt = CustomGroupLR(e_opt, decay_rates, intervals)
        s_opt = CustomGroupLR(e_opt, phase_settings)

        gt_landmark = landmarks

        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        all_train_iter = 0
        all_train_iters = []
        photometric_loss = []
        param_values_dict = {
            'shape': [],
            'exp': [],
            'pose': [],
            'cam': [],
            'tex': [],
            'latent_codes': [],
            'scale': [],
        }

        # specular_term = 0.5
        # nums = cfg.image_size * cfg.image_size

        # albedo = 1 - (torch.ones((nums, 3)) * specular_term) # N x 3
        # albedo_subset = albedo[image_masks] # K x 3
        # del albedo


        for k in range(cfg.max_iter + 500): # cfg.max_iter
            losses = {}

            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = util.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = util.batch_orth_proj(landmarks3d, cam)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]
            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2])

            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * cfg.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * cfg.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * cfg.w_pose_reg
            # losses['tex_reg'] = (torch.sum(tex ** 2) / 2) * 1e-4
            # losses['latent_code_reg'] = (torch.sum(latent_codes ** 2) / 2) * 1e-4

            if rigid_mode:
                losses['photometric_texture'] = torch.tensor(0.0)  # Set as a zero tensor
                losses['exponential'] = torch.tensor(0.0)  # Set as a zero tensor
            else:
                predicted_illumination = self.reni(rotation=None, latent_codes=latent_codes, scale=scale)
                # render
                albedos = self.flametex(tex) / 255.
                ops = self.render(vertices, trans_vertices, albedos, illumination=predicted_illumination)
                predicted_images = ops['images']
                # losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * config.w_pho

                losses['photometric_texture'] = F.smooth_l1_loss(image_masks * ops['images'],
                                                                image_masks * images) * cfg.w_pho

                losses['exponential'] = latent_codes.pow(2).mean() * 1e-4 * 10

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()
            s_opt.step()
            # if k % 100 == 0:
            #     s_opt.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

            # if k % 10 == 0:
            #     print(loss_info)


            if (k+1) % 50 == 0:
                if not rigid_mode:
                    # Get gradients (as a measure of contribution)
                    tex_grad = torch.norm(tex.grad).item()
                    latent_codes_grad = torch.norm(latent_codes.grad).item()
                    self.writer.add_scalar('Gradient/tex', tex_grad, k)
                    self.writer.add_scalar('Gradient/latent_codes', latent_codes_grad, k)

                all_train_iter += 10
                all_train_iters.append(all_train_iter)
                photometric_loss.append(losses['photometric_texture'])
                param_values_dict['shape'].append(shape.detach().cpu().numpy())
                param_values_dict['exp'].append(exp.detach().cpu().numpy())
                param_values_dict['pose'].append(pose.detach().cpu().numpy())
                param_values_dict['cam'].append(cam.detach().cpu().numpy())
                param_values_dict['tex'].append(tex.detach().cpu().numpy())
                param_values_dict['latent_codes'].append(latent_codes.detach().cpu().numpy())
                param_values_dict['scale'].append(scale.detach().cpu().numpy())
                # param_values_dict['lights'].append(lights.detach().cpu().numpy())

                self.writer.add_scalar('Loss/landmark', losses['landmark'].item(), k)
                self.writer.add_scalar('Loss/shape_reg', losses['shape_reg'].item(), k)
                self.writer.add_scalar('Loss/expression_reg', losses['expression_reg'].item(), k)
                self.writer.add_scalar('Loss/pose_reg', losses['pose_reg'].item(), k)
                self.writer.add_scalar('Loss/photometric_texture', losses['photometric_texture'].item(), k)
                self.writer.add_scalar('Loss/exponential', losses['exponential'].item(), k)
                self.writer.add_scalar('Loss/all_loss', all_loss.item(), k)

                self.writer.add_histogram('Parameters/shape', shape, k)
                self.writer.add_histogram('Parameters/exp', exp, k)
                self.writer.add_histogram('Parameters/pose', pose, k)
                self.writer.add_histogram('Parameters/cam', cam, k)
                self.writer.add_histogram('Parameters/tex', tex, k)
                self.writer.add_histogram('Parameters/latent_codes', latent_codes, k)
                self.writer.add_histogram('Parameters/scale', scale, k)

                # Log learning rates
                for i, group in enumerate(e_opt.param_groups):
                    self.writer.add_scalar(f'Learning Rate/Group {i+1}', group['lr'], k)

                print(loss_info)

                if not rigid_mode:
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        util.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                    grids['albedoimage'] = torchvision.utils.make_grid(
                        (ops['albedo_images'])[visind].detach().cpu())
                    illumination_image = F.interpolate(self.reni.visualize_illumination(predicted_illumination).permute(2, 0, 1).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    grids['illumination'] = torchvision.utils.make_grid(
                        (illumination_image).detach().cpu())
                    grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                    shape_images = self.render.render_shape(vertices, trans_vertices, images)
                    grids['shape'] = torchvision.utils.make_grid(
                        F.interpolate(shape_images[visind], [224, 224])).detach().float().cpu()

                    # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
                    del illumination_image, shape_images
                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                    # Convert numpy array to PIL Image for drawing text
                    pil_image = Image.fromarray(grid_image)

                    # Prepare to draw the iteration number
                    draw = ImageDraw.Draw(pil_image)
                    font = ImageFont.load_default() #= ImageFont.truetype("arial.ttf", 16)  # Specify the font and size you want
                    text = f"Iteration: {k}"

                    # Specify the position where you want the text
                    position = (10, 224*7 + 50)  # Change this as needed

                    # Draw text on the image
                    draw.text(position, text, font=font, fill=(255, 255, 255))  # White text or choose your color

                    # Convert back to numpy array if needed for video writing
                    grid_image = np.array(pil_image)

                    video_writer.write(grid_image)

                if k >= 500:
                    rigid_mode = False

        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'verts': trans_vertices.detach().cpu().numpy(),
            'albedos': albedos.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'latent_codes': latent_codes.detach().cpu().numpy(),
            'scale': scale.detach().cpu().numpy(),
        }

        self.writer.flush()

        # util.plot_all_parameter_heatmaps(param_values_dict, 100)

        # util.draw_train_process("training", all_train_iters, photometric_loss, 'photometric loss')
        # np.save("./test_results/model.npy", single_params)
        return single_params

    def run(self, img, net, rect_detect, landmark_detect, rect_thresh, save_name, video_writer, savefolder):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []
        image_masks = []
        bbox = rect_detect.extract(img, rect_thresh)
        if len(bbox) > 0:
            crop_image, new_bbox = util.crop_img(img, bbox[0], cfg.cropped_size)

            # input landmark
            resize_img, landmark = landmark_detect.extract([crop_image, [new_bbox]])
            landmark = landmark[0]
            landmark[:, 0] = landmark[:, 0] / float(resize_img.shape[1]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(resize_img.shape[0]) * 2 - 1
            landmarks.append(torch.from_numpy(landmark)[None, :, :].double().to(self.device))
            landmarks = torch.cat(landmarks, dim=0)

            # input image
            image = cv2.resize(crop_image, (cfg.cropped_size, cfg.cropped_size)).astype(np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            images.append(torch.from_numpy(image[None, :, :, :]).double().to(self.device))
            images = torch.cat(images, dim=0)
            images = F.interpolate(images, [cfg.image_size, cfg.image_size])

            # face segment mask
            image_mask = util.face_seg(crop_image, net, cfg.cropped_size)
            image_masks.append(torch.from_numpy(image_mask).double().to(cfg.device))
            image_masks = torch.cat(image_masks, dim=0)
            image_masks = F.interpolate(image_masks, [cfg.image_size, cfg.image_size])

            # check folder exist or not
            util.check_mkdir(savefolder)
            save_file = os.path.join(savefolder, save_name)

            # optimize
            single_params = self.optimize(images, landmarks, image_masks, video_writer)
            self.render.save_obj(filename=save_file,
                                 vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
                                 textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
                                 )
            np.save(save_file, single_params)


if __name__ == '__main__':
    image_path = str(sys.argv[1])
    device_name = str(sys.argv[2])

    save_name = os.path.split(image_path)[1].split(".")[0] + '.obj'
    save_video_name = os.path.split(image_path)[1].split(".")[0] + '.avi'
    video_writer = cv2.VideoWriter(os.path.join(cfg.save_folder, save_video_name),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 16,
                                   (cfg.image_size, cfg.image_size * 8))
    util.check_mkdir(cfg.save_folder)
    fitting = PhotometricFitting(device=device_name)
    img = cv2.imread(image_path)
    # padding_size = 800
    # img_padded = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)

    face_detect = detector.SFDDetector(device_name, cfg.rect_model_path)
    face_landmark = FAN_landmark.FANLandmarks(device_name, cfg.landmark_model_path, cfg.face_detect_type)

    seg_net = BiSeNet(n_classes=cfg.seg_class).cuda()
    seg_net.load_state_dict(torch.load(cfg.face_seg_model))
    seg_net.eval()

    fitting.run(img, seg_net, face_detect, face_landmark, cfg.rect_thresh, save_name, video_writer,
                cfg.save_folder)
