#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Configuration for libcamera cameras.

This module provides configuration options for cameras accessed through libcamera,
which is the standard camera interface for Raspberry Pi cameras.
"""

from dataclasses import dataclass

from ..configs import CameraConfig


@dataclass
@CameraConfig.register_subclass("libcamera")
class LibCameraCameraConfig(CameraConfig):
    """
    Configuration for libcamera cameras.
    
    This configuration is used for Raspberry Pi cameras that are accessed
    through the libcamera interface.
    """
    
    # Camera type identifier
    type: str = "libcamera"
    
    # Camera index (0 for first camera, 1 for second, etc.)
    camera_index: int = 0
    
    # Image width in pixels
    width: int = 1920
    
    # Image height in pixels  
    height: int = 1080
    
    # Frames per second (not used by libcamera-still but kept for compatibility)
    fps: int = 30
    
    # Quality setting for JPEG compression (1-100)
    quality: int = 90
    
    # Timeout for capturing images in milliseconds
    timeout_ms: int = 1000 