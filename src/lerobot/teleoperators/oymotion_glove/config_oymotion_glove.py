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

from dataclasses import dataclass, field
from typing import List

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("oymotion_glove")
@dataclass
class OyMotionGloveConfig(TeleoperatorConfig):
    """
    Configuration for OyMotion glove teleoperator.
    
    The OyMotion glove sends finger position data over UDP in the format:
    {
      "fingers": [0, 16384, 32768, 49152, 65535, 32768]
    }
    
    Values are 16-bit integers (0-65535) representing finger positions.
    """
    
    # UDP connection settings
    host: str = "10.33.10.154"
    port: int = 8889
    
    # Finger configuration
    finger_names: List[str] = field(default_factory=lambda: [
        "thumb",      # Index 0
        "index",      # Index 1
        "middle",     # Index 2
        "ring",       # Index 3
        "pinky",      # Index 4
        "extra"       # Index 5
    ])
    
    # Value ranges
    raw_min: int = 0        # Minimum raw value from glove
    raw_max: int = 65535    # Maximum raw value from glove (16-bit)
    
    # Timeout settings
    timeout_ms: float = 1000.0  # UDP receive timeout in milliseconds 