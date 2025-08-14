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

import logging
import socket
import json
import time
from datetime import datetime

from ..teleoperator import Teleoperator
from .config_zbot_leader import ZbotLeaderConfig

logger = logging.getLogger(__name__)


class ZbotLeader(Teleoperator):
    """
    Zbot Leader Arm - A 10-motor version using UDP communication
    """

    config_class = ZbotLeaderConfig
    name = "zbot_leader"

    def __init__(self, config: ZbotLeaderConfig):
        super().__init__(config)
        self.config = config
        
        # UDP socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Default joint positions (will be updated via UDP)
        # Using descriptive names matching zbot robot
        self.joint_positions = {}
        
        # Left arm joints (IDs 11-15)
        left_arm_names = [
            "left_shoulder_pitch",    # ID 11
            "left_shoulder_roll",     # ID 12  
            "left_shoulder_yaw",      # ID 13
            "left_elbow",             # ID 14
            "left_wrist"              # ID 15
        ]
        for name in left_arm_names:
            self.joint_positions[f"{name}.pos"] = 0.0
            
        # Right arm joints (IDs 21-25)
        right_arm_names = [
            "right_shoulder_pitch",   # ID 21
            "right_shoulder_roll",    # ID 22
            "right_shoulder_yaw",     # ID 23
            "right_elbow",            # ID 24
            "right_wrist"             # ID 25
        ]
        for name in right_arm_names:
            self.joint_positions[f"{name}.pos"] = 0.0
        self.last_update_time = 0
        self.connected = False

    @property
    def action_features(self) -> dict[str, type]:
        # Return joint features with descriptive names matching zbot robot
        features = {}
        # Left arm joints (IDs 11-15)
        left_arm_names = [
            "left_shoulder_pitch",    # ID 11
            "left_shoulder_roll",     # ID 12  
            "left_shoulder_yaw",      # ID 13
            "left_elbow",             # ID 14
            "left_wrist"              # ID 15
        ]
        for name in left_arm_names:
            features[f"{name}.pos"] = float
            
        # Right arm joints (IDs 21-25)
        right_arm_names = [
            "right_shoulder_pitch",   # ID 21
            "right_shoulder_roll",    # ID 22
            "right_shoulder_yaw",     # ID 23
            "right_elbow",            # ID 24
            "right_wrist"             # ID 25
        ]
        for name in right_arm_names:
            features[f"{name}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise Exception(f"{self} already connected")

        try:
            # Bind to UDP port
            self.sock.bind((self.config.host, self.config.port))
            self.connected = True
        except Exception as e:
            raise

    @property
    def is_calibrated(self) -> bool:
        # UDP-based teleoperator doesn't need calibration
        return True

    def calibrate(self) -> None:
        # UDP-based teleoperator doesn't need calibration
        logger.info("UDP-based teleoperator doesn't require calibration")
        pass

    def configure(self) -> None:
        # UDP-based teleoperator doesn't need configuration
        logger.info("UDP-based teleoperator doesn't require configuration")
        pass

    def setup_motors(self) -> None:
        # UDP-based teleoperator doesn't need motor setup
        logger.info("UDP-based teleoperator doesn't require motor setup")
        pass

    def get_action(self) -> dict[str, float]:
        # Simple UDP receive like test_pykos_fixed.py
        try:
            # Clear buffer to get most recent data
            self.sock.setblocking(False)
            latest_data = None
            latest_addr = None
            
            # Read all available packets, keep only the latest
            while True:
                try:
                    data, addr = self.sock.recvfrom(512)
                    latest_data = data
                    latest_addr = addr
                except socket.error:
                    break  # No more packets
            
            if latest_data is None:
                # No new data, return last known positions
                return self.joint_positions.copy()
                
            message = latest_data.decode('utf-8')
            joint_data = json.loads(message)
            
            # Extract joint positions
            joints = joint_data.get("joints", {})
            
            # Map joint IDs to descriptive names (like test_pykos_fixed.py)
            joint_id_to_name = {
                11: "left_shoulder_pitch",
                12: "left_shoulder_roll", 
                13: "left_shoulder_yaw",
                14: "left_elbow",
                15: "left_wrist",
                21: "right_shoulder_pitch",
                22: "right_shoulder_roll",
                23: "right_shoulder_yaw", 
                24: "right_elbow",
                25: "right_wrist"
            }
            
            # Update joint positions directly (no logging)
            for joint_id_str, position in joints.items():
                joint_id = int(joint_id_str)
                if joint_id in joint_id_to_name:
                    joint_name = joint_id_to_name[joint_id]
                    joint_key = f"{joint_name}.pos"
                    self.joint_positions[joint_key] = float(position)
            
            return self.joint_positions.copy()
            
        except socket.error as e:
            # No data available, return last known positions
            return self.joint_positions.copy()
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing UDP data: {e}")
            return self.joint_positions.copy()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback if needed
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        self.sock.close()
        self.connected = False