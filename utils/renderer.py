"""
Author: Yao Feng
Copyright (c) 2020, Yao Feng
All rights reserved.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from utils import util

from reni_plus_plus.model_components.shaders import LambertianShader
from reni_plus_plus.utils.colourspace import linear_to_sRGB


class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.

    Notice:
        x,y,z are in image space
    """

    def __init__(self, image_size=224):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        fixed_vetices = vertices.clone()
        fixed_vetices[..., :2] = -fixed_vetices[..., :2]
        meshes_screen = Meshes(verts=fixed_vetices.float(), faces=faces.long())
        raster_settings = self.raster_settings

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        # pix_to_face: (N, H, W, K) LongTensor giving the indices of the nearest face to each pixel.
        # zbuf: (N, H, W, K) FloatTensor giving the depth of the nearest face at each pixel.
        # bary_coords: (N, H, W, K, 3) FloatTensor giving the barycentric coordinates of the
        #             pixel within the face. This means that bary_coords[i, y, x, k] is the
        #             weight of the kth vertex of the pix_to_face[i, y, x, k]th face.
        # dists: (N, H, W, K) FloatTensor giving the euclidean distance of the pixel to the face.

        # Attribute Interpolation
        vismask = (pix_to_face > -1).float() # [N, H, W, K]
        D = attributes.shape[-1] # 3 for uv, 3 for normal, 3 for vertex, 3 for face normal
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1  # []
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        # import ipdb; ipdb.set_trace()
        return pixel_vals


class Renderer(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256, light_directions=None):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)

        # faces
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coordsw
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors
        colors = torch.tensor([74, 120, 168])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        # RENI feature flag
        if light_directions is not None:
            self.light_directions = light_directions
            self.lambertian_shader = LambertianShader("cuda")
        else:
            ## lighting
            pi = np.pi
            constant_factor = torch.tensor(
                [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))])
            self.register_buffer('constant_factor', constant_factor)



    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point', illumination=None, partial=False):
        '''
        lihgts:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
        vertices: [N, V, 3], vertices in work space, for calculating normals, then shading
        transformed_vertices: [N, V, 3], range(-1, 1), projected vertices, for rendering
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))

        # render
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(),
                                face_vertices.detach(), face_normals.detach()], -1)
        # import ipdb;ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]

        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # remove inner mouth region
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        if lights is not None:
            normal_images = rendering[:, 9:12, :, :].detach()
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  lights)
                    shading_images = shading.reshape(
                        [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1,
                                                                                                                  4, 2,
                                                                                                                  3)
                    shading_images = shading_images.mean(1)
                else:
                    shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      lights)
                    shading_images = shading.reshape(
                        [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1,
                                                                                                                  4, 2,
                                                                                                                  3)
                    shading_images = shading_images.mean(1)
            images = albedo_images * shading_images
        elif illumination is not None:
            normal_images = rendering[:, 9:12, :, :].detach()
            normal_images = normal_images.squeeze(0).permute(1, 2, 0) # C, H, W -> H, W, C
            normal_images = normal_images[..., [0,2,1]] # RGB -> BGR


            org_shape = normal_images.shape


            normal_images = normal_images / torch.norm(normal_images, dim=2, keepdim=True)

            mask = ~torch.isnan(normal_images)[..., 0] # Mask out the NaN values
            normal_images[~mask] = 0 # Set the NaN values to 0

            mask = mask.reshape(-1) # N = H x W
            if partial:
                mask = mask & (torch.rand(mask.shape).to(mask.device) < 0.25)
            normals_subet = normal_images.reshape(-1, 3)[mask] # K x 3

            albedo_image = albedo_images.squeeze(0).permute(1, 2, 0) # C, H, W -> H, W, C
            albedo_subset = albedo_image.reshape(-1, 3)[mask] # K x 3

            light_directions_subset = self.light_directions[mask]



            illumination_subset = illumination.unsqueeze(0).repeat(normals_subet.shape[0], 1, 1)

            predicted_sum, predicted_render = self.lambertian_shader(albedo=albedo_subset,
                                        normals=normals_subet,
                                        light_directions=light_directions_subset,
                                        light_colors=illumination_subset,
                                        detach_normals=True)


            # images = torch.zeros(og_shape[0] * og_shape[1], 3)
            # images[mask] = predicted_albedo
            # images = linear_to_sRGB(images.reshape(og_shape[0], og_shape[1], 3), use_quantile=True)
            # images = images.permute(2, 0, 1).unsqueeze(0)
            # images = images.to(alpha_images.device)

            # shading_images = predicted_render

            shading_images = torch.zeros(org_shape[0] * org_shape[1], 3).to(predicted_render.device)
            shading_images[mask] = predicted_render
            shading_images = linear_tonemap(shading_images)
            shading_images = shading_images.reshape(org_shape[0], org_shape[1], 3)
            shading_images = shading_images.permute(2, 0, 1).unsqueeze(0)
            images = shading_images

        else:
            images = albedo_images
            shading_images = images.detach() * 0.

        outputs = {
            'images': images * alpha_images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': rendering[:, 9:12, :, :].detach(),
        }

        if illumination is not None:
            outputs['mask'] = mask

        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1], \
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        # import ipdb; ipdb.set_trace()
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nlgiht, nv, 3]
        '''
        light_direction = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading

    def render_shape(self, vertices, transformed_vertices, images=None, lights=None):
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor([[-0.1, -0.1, 0.2],
                                            [0, 0, 1]]
                                           )[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1));
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1));
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        # render
        attributes = torch.cat(
            [self.face_colors.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(), face_vertices.detach(),
             face_normals.detach()], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        # albedo
        albedo_images = rendering[:, :3, :, :]
        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        if lights.shape[1] == 9:
            shading_images = self.add_SHlight(normal_images, lights)
        else:
            print('directional')
            shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)

            shading_images = shading.reshape(
                [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1, 4, 2, 3)
            shading_images = shading_images.mean(1)
        images = albedo_images * shading_images

        return images

    def render_normal(self, transformed_vertices, normals):
        '''
        -- rendering normal
        '''
        batch_size = normals.shape[0]

        # Attributes
        attributes = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(self, vertices):
        '''
        sample vertices from world space to uv space
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1)).clone().detach()
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]

        return uv_vertices

    def save_obj(self, filename, vertices, textures):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        util.save_obj(filename, vertices, self.faces[0], textures=textures, uvcoords=self.raw_uvcoords[0],
                      uvfaces=self.uvfaces[0])



def is_normalized(tensor):
    # Reshape to 2D tensor where each row is a 3-element vector
    reshaped_tensor = tensor.view(-1, 3)

    # Filter out rows where all elements are [0, 0, 0]
    non_zero_rows = reshaped_tensor[~torch.all(reshaped_tensor == 0, dim=1)]

    # Calculate the norm of each row
    norms = torch.norm(non_zero_rows, dim=1)

    # Check if all norms are close to 1 (within some tolerance, e.g., 1e-6)
    is_normalized = torch.allclose(norms, torch.tensor(1.0), atol=1e-6)

    # Output the result
    return is_normalized

# Tone Mapping using Reinhard Operator
def reinhard_tone_map(hdr, key=0.18):
    # Calculate the luminance (assuming RGB, use the Rec. 709 luminance coefficients)
    luminance = 0.2126 * hdr[:, 0] + 0.7152 * hdr[:, 1] + 0.0722 * hdr[:, 2]

    # Normalize the luminance
    L_avg = torch.mean(luminance)
    L_mapped = key * hdr / L_avg

    # Apply the Reinhard tone mapping operator
    tone_mapped = L_mapped / (1.0 + L_mapped)

    return tone_mapped

# Convert from linear to sRGB
def linear_to_srgb(linear):
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.pow(linear, 1/2.4) - 0.055
    )
    return srgb

def sRGB_to_linear(color, clamp=True):
    """Convert sRGB to linear RGB.

    Args:
        color: [..., 3]

        Returns:
            color: [..., 3]
    """

    color = torch.where(
        color <= 0.04045,
        color / 12.92,
        torch.pow((color + 0.055) / 1.055, 2.4),
    )

    if clamp:
        color = torch.clamp(color, 0.0, 1.0)
    return color


def tumblin_rushmeier_tone_mapping(input_tensor: torch.Tensor, Ld_max=1.0, Lw_max=1.0, k=0.18, delta=1e-6, gamma=2.2):
    """
    Apply complete Tumblin-Rushmeier tone mapping to an HDR image tensor with gamma correction.

    Args:
        input_tensor (torch.Tensor): The input HDR image tensor of shape [N, 3], where N is the number of pixels.
        Ld_max (float): Maximum display luminance. Default is 1.0.
        Lw_max (float): Maximum world luminance. Default is 1.0.
        k (float): Scaling factor. Default is 0.36 to make the image brighter.
        delta (float): Small constant to avoid logarithm of zero. Default is 1e-6.
        gamma (float): Gamma correction value. Default is 2.2.

    Returns:
        torch.Tensor: Tone-mapped LDR image tensor of shape [N, 3].
    """
    assert input_tensor.ndimension() == 2 and input_tensor.size(1) == 3, \
        "Input tensor must be of shape [N, 3] where N is the number of pixels and 3 represents RGB channels."

    # Convert RGB to luminance using standard luminance coefficients
    luminance_coefficients = torch.tensor([0.2126, 0.7152, 0.0722], device=input_tensor.device)
    Lw = torch.sum(input_tensor * luminance_coefficients, dim=1, keepdim=True)

    # Calculate the adaptation luminance (logarithmic average luminance)
    log_mean_luminance = torch.exp(torch.mean(torch.log(Lw + delta)))

    # Scale the luminance using the Tumblin-Rushmeier formula
    Ld = k * (Ld_max / Lw_max) * (Lw / log_mean_luminance)

    # Normalize the RGB values by the scaled luminance
    Lw_normalized = Ld / (Lw + delta)  # Adding delta to avoid division by zero
    output_tensor = input_tensor * Lw_normalized

    # Apply gamma correction to brighten the image
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
    output_tensor = torch.pow(output_tensor, 1.0 / gamma)

    return output_tensor


def linear_tonemap(hdr_pixels, gamma=2.2):
    # Ensure the tensor is in the correct format
    assert hdr_pixels.ndimension() == 2 and hdr_pixels.size(1) == 3, "Input tensor must be of shape [N, 3] where N is the number of pixels and 3 represents RGB channels."

    # Convert HDR pixels to float32 if they aren't already
    if hdr_pixels.dtype != torch.float32:
        hdr_pixels = hdr_pixels.float()

    # Find the minimum and maximum values in the pixels
    min_val = hdr_pixels.min()
    max_val = hdr_pixels.max()

    # Normalize the pixels to the [0, 1] range
    ldr_pixels = (hdr_pixels - min_val) / (max_val - min_val + 1e-5)  # Adding a small epsilon to avoid division by zero

    # Clamp values to avoid zero or negative values before gamma correction
    ldr_pixels = torch.clamp(ldr_pixels, min=1e-5)  # Clamping to a small positive value

    # Apply gamma correction
    ldr_pixels = torch.pow(ldr_pixels, 1.0 / gamma)

    # Clamp the output to the [0, 1] range
    ldr_pixels = torch.clamp(ldr_pixels, 0.0, 1.0)

    return ldr_pixels
