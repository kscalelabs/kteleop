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
import math
from datetime import datetime

from ..teleoperator import Teleoperator
from .config_oymotion_glove import OyMotionGloveConfig

logger = logging.getLogger(__name__)


class OyMotionGlove(Teleoperator):
    """
    OyMotion Glove Teleoperator - Receives finger position data over UDP
    
    Expected UDP data format:
    {
      "fingers": [0, 16384, 32768, 49152, 65535, 32768]
    }
    
    Where each value is a 16-bit integer (0-65535) representing finger position.
    """

    config_class = OyMotionGloveConfig
    name = "oymotion_glove"

    def __init__(self, config: OyMotionGloveConfig):
        super().__init__(config)
        self.config = config
        
        # UDP socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Initialize finger positions (normalized 0.0-1.0)
        self.finger_positions = {}
        for finger_name in self.config.finger_names:
            self.finger_positions[f"{finger_name}.pos"] = 0.0
        
        self.last_update_time = 0
        self.connected = False
        
        # Raw finger values for debugging
        self._raw_finger_values = [0] * 6

    @property
    def action_features(self) -> dict[str, type]:
        """Return finger features matching Inspire hand robot."""
        features = {}
        for finger_name in self.config.finger_names:
            features[f"{finger_name}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """No feedback features for OyMotion glove."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if the teleoperator is connected."""
        return self.connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to UDP port and start receiving glove data."""
        if self.is_connected:
            raise Exception(f"{self} already connected")

        try:
            # Bind to UDP port
            self.sock.bind((self.config.host, self.config.port))
            self.sock.settimeout(self.config.timeout_ms / 1000.0)  # Convert to seconds
            self.connected = True
            logger.info(f"OyMotion glove connected on {self.config.host}:{self.config.port}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect OyMotion glove: {e}")

    @property
    def is_calibrated(self) -> bool:
        """UDP-based teleoperator doesn't need calibration."""
        return True

    def calibrate(self) -> None:
        """UDP-based teleoperator doesn't need calibration."""
        logger.info("OyMotion glove doesn't require calibration")
        pass

    def configure(self) -> None:
        """UDP-based teleoperator doesn't need configuration."""
        logger.info("OyMotion glove doesn't require configuration")
        pass

    def setup_motors(self) -> None:
        """UDP-based teleoperator doesn't have motors."""
        logger.info("OyMotion glove doesn't have motors to setup")
        pass

    def _convert_udp_to_hand_value(self, raw_value: int) -> float:
        """Map UDP value (0-65535) to hand value (0-1000) using a normalized exponential curve.
        The curve passes through (0,0) and (65535,1000) with k = 3/65535.
        """
        # Clamp raw value to valid UDP range
        if raw_value < self.config.raw_min:
            raw_value = self.config.raw_min
        elif raw_value > self.config.raw_max:
            raw_value = self.config.raw_max

        # Exponential mapping: y = 1000 * (1 - exp(-k*x)) / (1 - exp(-k*65535))
        Xmax = 65535.0
        x = float(raw_value)
        k = 1.0 / Xmax
        denom = 1.0 - math.exp(-k * Xmax)
        if denom == 0.0:
            hand_value = 0.0
        else:
            hand_value = 1000.0 * (1.0 - math.exp(-k * x)) / denom

        # Clamp to [0, 1000]
        if hand_value < 0.0:
            hand_value = 0.0
        elif hand_value > 1000.0:
            hand_value = 1000.0
        return float(hand_value)
    


    def get_action(self) -> dict[str, float]:
        """Receive UDP data and convert to finger positions."""
        try:
            # Clear buffer to get most recent data
            self.sock.setblocking(False)
            latest_data = None
            latest_addr = None
            
            # Read all available packets, keep only the latest
            while True:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    latest_data = data
                    latest_addr = addr
                except socket.error:
                    break  # No more packets
            
            if latest_data is None:
                # No new data, return last known positions
                return self.finger_positions.copy()
                
            message = latest_data.decode('utf-8')
            glove_data = json.loads(message)
            
            # Extract finger data
            finger_values = glove_data.get("fingers", [])
            
            if len(finger_values) >= 6:
                # Store raw values for debugging
                self._raw_finger_values = finger_values[:6]
                
                # Convert UDP values directly to hand values
                for i, raw_value in enumerate(finger_values[:6]):
                    if i < len(self.config.finger_names):
                        finger_name = self.config.finger_names[i]
                        hand_value = self._convert_udp_to_hand_value(raw_value)
                        self.finger_positions[f"{finger_name}.pos"] = hand_value
                
                self.last_update_time = time.time()
                
                # Optional: Log received data for debugging
                logger.debug(f"Received glove data: raw={self._raw_finger_values}, "
                           f"converted={[self.finger_positions.get(f'{name}.pos', 0) for name in self.config.finger_names]}")
                
                return self.finger_positions.copy()
            else:
                logger.warning(f"Insufficient finger data received: {len(finger_values)} values")
                return self.finger_positions.copy()
                
        except socket.error as e:
            # No data available, return last known positions
            return self.finger_positions.copy()
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing glove data: {e}")
            return self.finger_positions.copy()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """OyMotion glove doesn't support force feedback."""
        # Could potentially send feedback data back to glove if supported
        pass

    def get_debug_info(self) -> dict:
        """Get debug information about glove state."""
        return {
            "raw_values": self._raw_finger_values,
            "normalized_positions": self.finger_positions.copy(),
            "last_update": self.last_update_time,
            "connected": self.connected
        }

    def disconnect(self) -> None:
        """Disconnect from UDP port."""
        if not self.is_connected:
            return

        if self.sock:
            self.sock.close()
        
        self.connected = False
        logger.info("Disconnected from OyMotion glove") 