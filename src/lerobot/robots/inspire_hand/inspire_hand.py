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

import asyncio
import logging
import serial
import struct
import time
from functools import cached_property
from typing import Any, Dict, List

import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_inspire_hand import InspireHandConfig

logger = logging.getLogger(__name__)


class InspireHand(Robot):
    """
    Inspire hand robot interface.
    
    This robot class provides control over the Inspire 6-finger robotic hand.
    It communicates via serial port using the Inspire protocol.
    Each finger can be controlled independently with position values from 0-1000.
    """

    config_class = InspireHandConfig
    name = "inspire_hand"

    def __init__(self, config: InspireHandConfig):
        super().__init__(config)
        self.config = config
        self.serial_port = None
        self._connected = False
        self._calibrated = False
        
        # Store current finger positions
        self._current_positions = {finger_id: 0 for finger_id in self.config.finger_ids}
        self._last_update_time = 0.0
        
        # Create finger ID to name mappings
        self.finger_id_to_name = {}
        for finger_id, name in zip(self.config.finger_ids, self.config.finger_names):
            self.finger_id_to_name[finger_id] = name

    @property
    def _finger_features(self) -> dict[str, type]:
        """Return feature types for all fingers."""
        features = {}
        for finger_id in self.config.finger_ids:
            finger_name = self.finger_id_to_name[finger_id]
            features[f"{finger_name}.pos"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type]:
        """Define the structure of observations returned by the robot."""
        return self._finger_features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define the structure of actions accepted by the robot."""
        features = {}
        for finger_id in self.config.finger_ids:
            finger_name = self.finger_id_to_name[finger_id]
            features[f"{finger_name}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        """Whether the robot is currently connected."""
        return self._connected

    def _write_register(self, addr: int, values: List[int]) -> None:
        """Write to Inspire hand register."""
        if self.serial_port is None or not getattr(self.serial_port, "is_open", False):
            raise DeviceNotConnectedError("Serial port not open")
        
        # Create command packet
        num_values = len(values)
        bytes_data = []
        for val in values:
            bytes_data.extend([val & 0xFF, (val >> 8) & 0xFF])
        
        # Command: [0xEB, 0x90, ID, LEN, 0x12, ADDR_L, ADDR_H, DATA..., CHECKSUM]
        cmd = [0xEB, 0x90, self.config.hand_id, num_values * 2 + 3, 0x12]
        cmd.extend([addr & 0xFF, (addr >> 8) & 0xFF])
        cmd.extend(bytes_data)
        
        # Calculate checksum
        checksum = sum(cmd[2:]) & 0xFF
        cmd.append(checksum)
        
        # Send command
        self.serial_port.write(bytes(cmd))
        time.sleep(0.005)  # 10ms delay
        
        # Clear response buffer
        while self.serial_port.in_waiting:
            self.serial_port.read()

    def _read_register(self, addr: int, num_bytes: int) -> List[int]:
        """Read from Inspire hand register - optimized for speed using exact protocol."""
        if self.serial_port is None or not getattr(self.serial_port, "is_open", False):
            raise DeviceNotConnectedError("Serial port not open")
        
        # Create read command according to protocol
        cmd = [0xEB, 0x90, self.config.hand_id, 0x04, 0x11]
        cmd.extend([addr & 0xFF, (addr >> 8) & 0xFF, num_bytes])
        
        # Calculate checksum
        checksum = sum(cmd[2:]) & 0xFF
        cmd.append(checksum)
        
        # Send command
        self.serial_port.write(bytes(cmd))
        
        # Read exact response size: header(2) + ID(1) + length(1) + flag(1) + addr(2) + data(12) + checksum(1) = 20 bytes
        expected_size = 8 + num_bytes  # Protocol overhead + data
        response = self.serial_port.read(expected_size)
        
        if len(response) < expected_size:
            raise RuntimeError(f"Incomplete data: got {len(response)}, expected {expected_size}")
        
        # Validate header: 0x90, 0xEB (note: response header is reversed!)
        if response[0] != 0x90 or response[1] != 0xEB:
            raise RuntimeError("Invalid response header")
        
        # Extract data directly (skip parsing overhead)
        # Data starts at byte[7] and is num_bytes long
        values = []
        for i in range(0, num_bytes, 2):
            if 7 + i + 1 < len(response):
                # Little-endian: low byte first, then high byte
                val = response[7 + i] | (response[7 + i + 1] << 8)
                values.append(val)
        
        return values

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the Inspire hand via serial port."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            # Open serial port - Reliable settings
            self.serial_port = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baudrate,
                timeout=0.025,  # 25ms read timeout - fast but reliable
                write_timeout=0.05  # 50ms write timeout - more reliable for commands
            )
            
            # Test communication by reading current positions
            try:
                positions = self._read_register(1546, 12)  # angleAct register
                if len(positions) >= 6:
                    for i, pos in enumerate(positions[:6]):
                        self._current_positions[i] = pos
                    logger.info(f"Successfully connected to Inspire hand on {self.config.serial_port}")
                    self._connected = True
                else:
                    raise RuntimeError("Invalid response from hand")
            except Exception as e:
                self.serial_port.close()
                raise RuntimeError(f"Failed to communicate with hand: {e}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Inspire hand: {e}")

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is calibrated."""
        return self._calibrated

    def calibrate(self) -> None:
        """Calibrate the Inspire hand."""
        if not self._connected:
            raise DeviceNotConnectedError("Hand not connected")
        
        try:
            # Read current positions to establish baseline
            positions = self._read_register(1546, 12)  # angleAct register
            if len(positions) >= 6:
                for i, pos in enumerate(positions[:6]):
                    self._current_positions[i] = pos
                self._calibrated = True
                logger.info("Inspire hand calibrated successfully")
            else:
                raise RuntimeError("Failed to read finger positions during calibration")
        except Exception as e:
            raise RuntimeError(f"Calibration failed: {e}")

    def configure(self) -> None:
        """Configure the Inspire hand."""
        if not self._connected:
            raise DeviceNotConnectedError("Hand not connected")
        
        # No specific configuration needed for Inspire hand
        logger.info("Inspire hand configured")

    def get_observation(self) -> dict[str, Any]:
        """Get current finger positions from the hand."""
        if not self._connected:
            raise DeviceNotConnectedError("Hand not connected")
        
        try:
            # Read current positions
            positions = self._read_register(1546, 12)  # angleAct register
            if len(positions) >= 6:
                observation = {}
                for i, pos in enumerate(positions[:6]):
                    finger_name = self.finger_id_to_name[i]
                    # Convert from hand units (0-1000) to normalized position (0.0-1.0)
                    normalized_pos = pos / 1000.0
                    observation[f"{finger_name}.pos"] = normalized_pos
                    self._current_positions[i] = pos
                self._last_update_time = time.time()
                return observation
            else:
                raise RuntimeError("Invalid position data received")
        except Exception as e:
            logger.error(f"Failed to get observation: {e}")
            # Return last known positions
            observation = {}
            for finger_id in self.config.finger_ids:
                finger_name = self.finger_id_to_name[finger_id]
                pos = self._current_positions[finger_id]
                normalized_pos = pos / 1000.0
                observation[f"{finger_name}.pos"] = normalized_pos
            return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send finger position commands to the hand."""
        if not self._connected:
            raise DeviceNotConnectedError("Hand not connected")
        
        try:
            # Prepare finger positions array
            finger_positions = [0] * 6
            
            # Process each finger command
            for finger_name, hand_pos in action.items():
                if finger_name.endswith('.pos'):
                    base_name = finger_name[:-4]  # Remove '.pos'
                    if base_name in self.config.finger_names:
                        finger_idx = self.config.finger_names.index(base_name)
                        # Values are already in hand units (0-1000), just clamp
                        hand_pos = int(hand_pos)
                        hand_pos = max(self.config.min_position, min(self.config.max_position, hand_pos))
                        finger_positions[finger_idx] = hand_pos
            
            # Send position command
            self._write_register(1486, finger_positions)  # angleSet register
            
            return action
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from the Inspire hand."""
        if not self._connected:
            return
        
        if self.serial_port:
            self.serial_port.close()
        
        self._connected = False
        self._calibrated = False
        logger.info("Disconnected from Inspire hand") 