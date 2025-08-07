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
from typing import Dict, List

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("zbot")
@dataclass
class ZBotConfig(RobotConfig):
    # KOS connection parameters
    host: str = "localhost"
    port: int = 50051
    
    # Joint configuration for upper body
    left_arm_ids: List[int] = field(default_factory=lambda: [11, 12, 13, 14, 15])
    right_arm_ids: List[int] = field(default_factory=lambda: [21, 22, 23, 24, 25])
    
    # Joint names for better identification
    left_arm_names: List[str] = field(default_factory=lambda: [
        "left_shoulder_pitch",    # ID 11
        "left_shoulder_roll",     # ID 12  
        "left_shoulder_yaw",      # ID 13
        "left_elbow",             # ID 14
        "left_wrist"              # ID 15
    ])
    
    right_arm_names: List[str] = field(default_factory=lambda: [
        "right_shoulder_pitch",   # ID 21
        "right_shoulder_roll",    # ID 22
        "right_shoulder_yaw",     # ID 23
        "right_elbow",            # ID 24
        "right_wrist"             # ID 25
    ])
    
    # Safety limits (in radians)
    max_velocity: float = 7200.0  # degrees per second converted to rad/s
    max_position_change: float = 2.0  # radians per command (increased for larger movements)
    
    # Control parameters
    command_rate_hz: float = 100.0
    update_rate_hz: float = 100.0
    
    # cameras (optional)
    cameras: dict[str, CameraConfig] = field(default_factory=dict) 