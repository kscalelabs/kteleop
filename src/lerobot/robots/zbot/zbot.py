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
import math
import time
from functools import cached_property
from typing import Any, Dict, List

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.constants import OBS_STATE
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import MotorCalibration

from ..robot import Robot
from .config_zbot import ZBotConfig

logger = logging.getLogger(__name__)


class ZBot(Robot):
    """
    ZBot robot interface for kos-kbot platform.
    
    This robot class provides control over the upper body joints (arms) of the kos-kbot.
    It interfaces with the KOS platform using the pykos Python client.
    """

    config_class = ZBotConfig
    name = "zbot"

    def __init__(self, config: ZBotConfig):
        super().__init__(config)
        self.config = config
        self.kos_client = None
        self._connected = False
        
        # Create joint ID to name mappings
        self.joint_id_to_name = {}
        for joint_id, name in zip(self.config.left_arm_ids, self.config.left_arm_names):
            self.joint_id_to_name[joint_id] = name
        for joint_id, name in zip(self.config.right_arm_ids, self.config.right_arm_names):
            self.joint_id_to_name[joint_id] = name
            
        # All joint IDs for easy access
        self.all_joint_ids = self.config.left_arm_ids + self.config.right_arm_ids
        
        # Initialize cameras if configured
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Store current joint states
        self._current_positions = {}
        self._current_velocities = {}
        self._last_update_time = 0.0

    @property
    def _joint_features(self) -> dict[str, type]:
        """Return feature types for all joints."""
        features = {}
        for joint_id in self.all_joint_ids:
            joint_name = self.joint_id_to_name[joint_id]
            features[f"{joint_name}.pos"] = float
            features[f"{joint_name}.vel"] = float
        return features

    @property
    def _camera_features(self) -> dict[str, tuple]:
        """Return feature types for cameras."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define the structure of observations returned by the robot."""
        return {**self._joint_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define the structure of actions accepted by the robot."""
        features = {}
        for joint_id in self.all_joint_ids:
            joint_name = self.joint_id_to_name[joint_id]
            features[f"{joint_name}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        """Check if the robot is connected to KOS."""
        return self._connected and self.kos_client is not None

    async def _connect_kos(self) -> None:
        """Establish connection to KOS platform."""
        try:
            import pykos
            # Create KOS client directly (sync API)
            self.kos_client = pykos.KOS()
            
            # Configure all actuators for control
            for joint_id in self.all_joint_ids:
                # Configure for control (like test_pykos_fixed.py)
                self.kos_client.actuator.configure_actuator(
                    actuator_id=joint_id,
                    kp=10,  # Position gain (reduced from 150)
                    kd=1,   # Velocity gain (reduced from 10)
                    torque_enabled=True  # Enable torque control
                )
            
        except ImportError:
            raise ImportError("pykos is required to use ZBot. Install with: pip install pykos")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to KOS: {e}")

    def connect(self, calibrate: bool = True) -> None:
        """Establish communication with the robot - simplified."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to KOS
        asyncio.run(self._connect_kos())
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        """Check if the robot is calibrated."""
        # For ZBot, we consider it calibrated if we have calibration data
        return len(self.calibration) > 0

    def calibrate(self) -> None:
        """Calibrate the robot by collecting joint offsets."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        logger.info("Starting ZBot calibration...")
        
        # Get current joint positions as calibration offsets
        observation = asyncio.run(self._get_joint_states())
        
        for joint_id in self.all_joint_ids:
            joint_name = self.joint_id_to_name[joint_id]
            if f"{joint_name}.pos" in observation:
                offset = observation[f"{joint_name}.pos"]
                # Create a simple calibration entry (MotorCalibration expects different parameters)
                self.calibration[joint_name] = MotorCalibration(
                    id=joint_id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=-180,
                    range_max=180
                )
                logger.info(f"Calibrated {joint_name} with offset: {offset:.4f}")
        
        # Save calibration data
        self._save_calibration()
        logger.info("ZBot calibration completed")

    def configure(self) -> None:
        """Apply runtime configuration to the robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        logger.info("Configuring ZBot...")
        
        # Set control parameters for all joints
        for joint_id in self.all_joint_ids:
            joint_name = self.joint_id_to_name[joint_id]
            # Store current position as reference
            self._current_positions[joint_id] = 0.0
            self._current_velocities[joint_id] = 0.0
            
        logger.info("ZBot configuration completed")

    async def _get_joint_states(self) -> dict[str, Any]:
        """Get current joint states from KOS."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        try:
            # Get actuator states from KOS (sync call)
            state_list = self.kos_client.actuator.get_actuators_state(self.all_joint_ids)
            
            observation = {}
            for actuator_state in state_list:
                joint_id = actuator_state.actuator_id
                joint_name = self.joint_id_to_name[joint_id]
                
                # Store position and velocity
                observation[f"{joint_name}.pos"] = actuator_state.position
                observation[f"{joint_name}.vel"] = actuator_state.velocity
                
                # Update internal state
                self._current_positions[joint_id] = actuator_state.position
                self._current_velocities[joint_id] = actuator_state.velocity
                
            self._last_update_time = time.time()
            return observation
            
        except Exception as e:
            logger.error(f"Failed to get joint states: {e}")
            # Return last known state if available
            observation = {}
            for joint_id in self.all_joint_ids:
                joint_name = self.joint_id_to_name[joint_id]
                observation[f"{joint_name}.pos"] = self._current_positions.get(joint_id, 0.0)
                observation[f"{joint_name}.vel"] = self._current_velocities.get(joint_id, 0.0)
            return observation

    def get_observation(self) -> dict[str, Any]:
        """Retrieve the current observation from the robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        # Get joint states
        observation = asyncio.run(self._get_joint_states())
        
        # Add camera observations if available
        for camera_name, camera in self.cameras.items():
            if camera.is_connected:
                try:
                    image = camera.get_image()
                    if image is not None:
                        observation[camera_name] = image
                except Exception as e:
                    logger.warning(f"Failed to get image from camera {camera_name}: {e}")
                    
        return observation

    def _send_joint_commands(self, action: dict[str, float]) -> dict[str, float]:
        """Send joint commands to KOS - simplified like test_pykos_fixed.py."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        try:
            # Create actuator commands directly (like test_pykos_fixed.py)
            actuator_commands = []
            
            for joint_id in self.all_joint_ids:
                joint_name = self.joint_id_to_name[joint_id]
                action_key = f"{joint_name}.pos"
                
                if action_key in action:
                    position = action[action_key]
                    command = {
                        'actuator_id': joint_id,
                        'position': position
                    }
                    actuator_commands.append(command)
            
            # Send commands directly (no logging, no safety limits)
            if actuator_commands:
                self.kos_client.actuator.command_actuators(actuator_commands)
                
            return action
            
        except Exception as e:
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action command to the robot - simplified."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
                
        # Send commands directly (no logging)
        result = self._send_joint_commands(action)
        return result

    def disconnect(self) -> None:
        """Disconnect from the robot and perform cleanup."""
        if not self.is_connected:
            return
            
        # Disconnect cameras
        for camera in self.cameras.values():
            camera.disconnect()
            
        # Disconnect from KOS
        if self.kos_client is not None:
            try:
                # Note: We don't need to explicitly call __aexit__ since we're not using context manager
                # The connection will be cleaned up automatically
                pass
            except Exception as e:
                pass
            finally:
                self.kos_client = None
                
        self._connected = False 