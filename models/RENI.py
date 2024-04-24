import os
import re
from pathlib import Path
import torch
import yaml

from reni_plus_plus.illumination_fields.reni_illumination_field import RENIFieldConfig
from reni_plus_plus.model_components.illumination_samplers import EquirectangularSamplerConfig
from reni_plus_plus.field_components.field_heads import RENIFieldHeadNames
from reni_plus_plus.utils.colourspace import linear_to_sRGB

class RENI:
    # def __init__(self, latent_dim_size=100, ckpt_step=50000):
    # TODO: Maybe more stuff could be moved to config
    # TODO: Add more functions to do trivial things
    def __init__(self, config):
        self.latent_dim_size = config.reni_latent_dim_size
        self.ckpt_step = str(config.nerfstudio_ckpt_step).zfill(9)  # Pads to 9 digits
        self.base_path = Path(f"model/reni_plus_plus_models/latent_dim_{self.latent_dim_size}")
        self.config_path = self.base_path / 'config.yml'
        self.ckpt_path = self.base_path / 'nerfstudio_models' / f'step-{self.ckpt_step}.ckpt'
        self.config = self.load_config()
        self.model = self.load_model()
        self.pixel_num = config.image_size * config.image_size
        self.width = config.reni_env_map_width
        self.sampler = EquirectangularSampler(self.width, False, False, self.pixel_num)

    def __call__(self, rotation, latent_codes, scale):
        return self.use_model(rotation, latent_codes, scale)

    def clean_and_load_yaml(self, yaml_content):
        cleaned_content = re.sub(r'!!python[^\s]*', '', yaml_content)
        return yaml.safe_load(cleaned_content)

    def load_config(self):
        with open(self.config_path, 'r') as f:
            content = f.read()
        return self.clean_and_load_yaml(content)['pipeline']['model']['field']

    def load_model(self):
        ckpt = torch.load(self.ckpt_path, map_location='cpu')
        model_config = RENIFieldConfig(
            axis_of_invariance=self.config['axis_of_invariance'],
            conditioning=self.config['conditioning'],
            encoded_input=self.config['encoded_input'],
            equivariance=self.config['equivariance'],
            first_omega_0=self.config['first_omega_0'],
            fixed_decoder=True,  # We are using the pre-trained decoder
            hidden_features=self.config['hidden_features'],
            hidden_layers=self.config['hidden_layers'],
            hidden_omega_0=self.config['hidden_omega_0'],
            invariant_function=self.config['invariant_function'],
            last_layer_linear=self.config['last_layer_linear'],
            latent_dim=self.config['latent_dim'],
            mapping_features=self.config['mapping_features'],
            mapping_layers=self.config['mapping_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            num_attention_layers=self.config['num_attention_layers'],
            old_implementation=self.config['old_implementation'],
            out_features=self.config['out_features'],
            output_activation=self.config['output_activation'],
            positional_encoding=self.config['positional_encoding'],
            trainable_scale=self.config['trainable_scale'],
        )

        model = model_config.setup(
            num_train_data=None,  # None as we are only using decoder
            num_eval_data=None,  # None as we are only using decoder
            normalisations={
                "min_max": ckpt['pipeline']['_model.field.min_max'].item(),
                "log_domain": ckpt['pipeline']['_model.field.log_domain'].item()
            }
        )

        model_dict = {}
        match_str = "_model.field."
        ignore_strs = [
            "_model.field.train_logvar",
            "_model.field.eval_logvar",
            "_model.field.train_mu",
            "_model.field.eval_mu",
        ]
        for key in ckpt["pipeline"].keys():
            if key.startswith(match_str) and not any(ignore_str in key for ignore_str in ignore_strs):
                model_dict[key[len(match_str):]] = ckpt["pipeline"][key]

        # Load weights of the decoder
        model.load_state_dict(model_dict, strict=False)
        model = model.to(torch.device("cpu"))
        return model

    def use_model(self, rotation, latent_codes, scale):
        z = latent_codes.repeat(self.sampler.get_num_rays(), 1, 1)  # [D, latent_dim, 3]
        s = scale.repeat(self.sampler.get_num_rays()) # [D]
        reni_output = self.model(ray_samples=self.sampler.get_ray_samples(), rotation=rotation, latent_codes=z, scale=s)
        predicted_illumination = reni_output[RENIFieldHeadNames.RGB.value]
        predicted_illumination = self.model.unnormalise(predicted_illumination)
        predicted_illumination = linear_to_sRGB(predicted_illumination, use_quantile=True)

        # predicted_illumination = predicted_illumination.unsqueeze(0).repeat(self.pixel_num, 1, 1)
        return predicted_illumination

    def to(self, device):
        self.model = self.model.to(device)
        self.sampler = self.sampler.to(device)
        return self

    def unnormalise(self, tensor):
        return self.model.unnormalise(tensor)

    def get_light_directions(self):
        return self.sampler.get_light_directions()

    def visualize_illumination(self, illumination):
        return illumination.reshape(self.width // 2, self.width, 3)

class EquirectangularSampler:
    def __init__(self, width, apply_random_rotation, remove_lower_hemisphere, pixel_num=4096):
        self.width = width
        self.apply_random_rotation = apply_random_rotation
        self.remove_lower_hemisphere = remove_lower_hemisphere

        self.direction_sampler = self.setup()

        self.ray_samples = self.generate_direction_samples()
        self.num_rays = self.get_num_rays()
        self.light_directions = self.generate_light_directions(pixel_num)

    def __call__(self):
        return self.get_ray_samples(), self.get_light_directions(), self.get_num_rays()

    def setup(self):
        direction_sampler = EquirectangularSamplerConfig(
            width=self.width,
            apply_random_rotation=self.apply_random_rotation,
            remove_lower_hemisphere=self.remove_lower_hemisphere
        ).setup()

        return direction_sampler

    def generate_direction_samples(self):
        ray_samples = self.direction_sampler.generate_direction_samples() # D
        ray_samples.directions = ray_samples.directions.reshape(-1, 3)  # D x 3
        return ray_samples

    def generate_light_directions(self, pixel_num):
        light_directions = self.ray_samples.directions.unsqueeze(0).repeat(pixel_num, 1, 1)
        return light_directions

    def get_ray_samples(self):
        return self.ray_samples

    def get_light_directions(self):
        return self.light_directions

    def get_num_rays(self):
        return self.ray_samples.directions.shape[0]

    def to(self, device):
        # self.direction_sampler = self.direction_sampler.to(device)
        self.ray_samples.directions = self.ray_samples.directions.to(device)
        return self


