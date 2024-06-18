# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Samplers for spherical illumination fields.
"""

from abc import abstractmethod
from typing import Optional, Type
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from reni_plus_plus.configs.base_config import InstantiateConfig
from reni_plus_plus.cameras.rays import RaySamples


# Field related configs
@dataclass
class IlluminationSamplerConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: IlluminationSampler)
    """target class to instantiate"""


class IlluminationSampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        config: IlluminationSamplerConfig,
    ) -> None:
        super().__init__()

    @abstractmethod
    def generate_direction_samples(self, num_directions: Optional[int] = None, apply_random_rotation=None) -> torch.Tensor:
        """Generate Direction Samples"""

    def forward(self, num_directions: Optional[int] = None, apply_random_rotation=None) -> torch.Tensor:
        """Returns directions for each position.

        Args:
            num_directions: number of directions to sample

        Returns:
            directions: [num_directions, 3]
        """

        return self.generate_direction_samples(num_directions, apply_random_rotation)


# Field related configs
@dataclass
class EquirectangularSamplerConfig(IlluminationSamplerConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: EquirectangularSampler)
    """target class to instantiate"""
    width: int = 256
    """width of the equirectangular image"""
    apply_random_rotation: bool = False
    """apply random rotation to the directions"""
    remove_lower_hemisphere: bool = False
    """remove lower hemisphere"""


class EquirectangularSampler(IlluminationSampler):
    """For sampling directions for an equirectangular image."""

    def __init__(
        self,
        config: EquirectangularSamplerConfig,
    ):
        super().__init__(config)
        self._width = config.width
        self.height = self._width // 2
        self.apply_random_rotation = config.apply_random_rotation
        self.remove_lower_hemisphere = config.remove_lower_hemisphere

    def generate_direction_samples(self, num_directions=None, apply_random_rotation=None, mask=None) -> RaySamples:

        if num_directions is None or num_directions == self.height * self._width:
            sidelen = self._width
            u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
            v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
            v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
            uv = torch.stack((u_grid, v_grid), -1)  # [sidelen/2, sidelen, 2]
            uv = uv.reshape(-1, 2)  # [sidelen/2*sidelen,2]
            theta = np.pi * (uv[:, 0] - 1)
            phi = np.pi * uv[:, 1]
            directions = torch.stack(
                (
                    torch.sin(phi) * torch.sin(theta),
                    -torch.sin(phi) * torch.cos(theta),
                    torch.cos(phi),
                ),
                -1,
            )

            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask.reshape(-1)
                directions = directions[mask] # [num_directions, 3]
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            # Sampling theta and phi
            theta_rand = 2 * torch.pi * torch.rand(num_directions)
            phi_rand = torch.acos(1 - 2 * torch.rand(num_directions))

            # Convert to Cartesian coordinates
            x = torch.sin(phi_rand) * torch.cos(theta_rand)
            y = torch.sin(phi_rand) * torch.sin(theta_rand)
            z = torch.cos(phi_rand)

            # Map to equirectangular image coordinates
            u = ((theta_rand / (2 * torch.pi)) * self._width).long()
            v = ((phi_rand / torch.pi) * self.height).long()

            # Unit norm directions
            directions = torch.stack([x, y, z], dim=1)


        ray_samples = RaySamples(directions=directions,
                                 camera_indices=torch.zeros_like(directions[:, 0:1]).long())

        return ray_samples
