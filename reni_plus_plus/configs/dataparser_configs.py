# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Aggregate all the dataparser configs in one location.
"""

from typing import TYPE_CHECKING

import tyro

from reni_plus_plus.data.dataparsers.arkitscenes_dataparser import ARKitScenesDataParserConfig
from reni_plus_plus.data.dataparsers.base_dataparser import DataParserConfig
from reni_plus_plus.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from reni_plus_plus.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from reni_plus_plus.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from reni_plus_plus.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from reni_plus_plus.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from reni_plus_plus.data.dataparsers.minimal_dataparser import MinimalDataParserConfig
from reni_plus_plus.data.dataparsers.nerfosr_dataparser import NeRFOSRDataParserConfig
from reni_plus_plus.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from reni_plus_plus.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from reni_plus_plus.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from reni_plus_plus.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig
from reni_plus_plus.data.dataparsers.scannetpp_dataparser import ScanNetppDataParserConfig
from reni_plus_plus.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from reni_plus_plus.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from reni_plus_plus.plugins.registry_dataparser import discover_dataparsers

dataparsers = {
    "nerfstudio-data": NerfstudioDataParserConfig(),
    "minimal-parser": MinimalDataParserConfig(),
    "arkit-data": ARKitScenesDataParserConfig(),
    "blender-data": BlenderDataParserConfig(),
    "instant-ngp-data": InstantNGPDataParserConfig(),
    "nuscenes-data": NuScenesDataParserConfig(),
    "dnerf-data": DNeRFDataParserConfig(),
    "phototourism-data": PhototourismDataParserConfig(),
    "dycheck-data": DycheckDataParserConfig(),
    "scannet-data": ScanNetDataParserConfig(),
    "sdfstudio-data": SDFStudioDataParserConfig(),
    "nerfosr-data": NeRFOSRDataParserConfig(),
    "sitcoms3d-data": Sitcoms3DDataParserConfig(),
    "scannetpp-data": ScanNetppDataParserConfig(),
    "colmap": ColmapDataParserConfig(),
}

external_dataparsers, _ = discover_dataparsers()
all_dataparsers = {**dataparsers, **external_dataparsers}

if TYPE_CHECKING:
    # For static analysis (tab completion, type checking, etc), just use the base
    # dataparser config.
    DataParserUnion = DataParserConfig
else:
    # At runtime, populate a Union type dynamically. This is used by `tyro` to generate
    # subcommands in the CLI.
    DataParserUnion = tyro.extras.subcommand_type_from_defaults(
        all_dataparsers,
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[DataParserUnion]  # Omit prefixes of flags in subcommands.
"""Union over possible dataparser types, annotated with metadata for tyro. This is
the same as the vanilla union, but results in shorter subcommand names."""