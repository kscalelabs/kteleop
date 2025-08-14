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

from dataclasses import dataclass
from typing import Dict, Any

from ..config import RobotConfig


@RobotConfig.register_subclass("inspire_hand")
@dataclass
class InspireHandConfig(RobotConfig):
    """
    Configuration for the Inspire hand robot.
    
    The Inspire hand is a 6-finger robotic hand that communicates via serial port.
    Each finger can be controlled independently with position values from 0-1000.
    """
    
    # Serial communication settings
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    hand_id: int = 1
    
    # Finger configuration
    finger_names: list[str] = None  # Will be set in __post_init__
    finger_ids: list[int] = None    # Will be set in __post_init__
    
    # Position limits (0 = closed, 1000 = open)
    min_position: int = 0
    max_position: int = 1000
    
    # Polling settings
    poll_rate_hz: float = 20.0  # 50ms intervals
    
    def __post_init__(self):
        """Initialize finger names and IDs if not provided."""
        if self.finger_names is None:
            self.finger_names = [
                "pinky",      # Finger 0 (was thumb)
                "ring",       # Finger 1 (was index)
                "middle",     # Finger 2 (unchanged)
                "index",      # Finger 3 (was ring)
                "thumb",      # Finger 4 (was pinky)
                "extra"       # Finger 5 (unchanged)
            ]
        
        if self.finger_ids is None:
            self.finger_ids = list(range(6))  # 0-5 for 6 fingers 