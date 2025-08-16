#!/usr/bin/env python

import json
import logging
import math
import socket
import time
from typing import Any

from ..teleoperator import Teleoperator
from .config_zbot_inspire_combined import ZBotInspireCombinedConfig

logger = logging.getLogger(__name__)


class ZBotInspireCombined(Teleoperator):
    """
    Combined ZBot + Inspire hand teleoperator that receives both joint and finger data
    over a single UDP port in one packet.
    """

    config_class = ZBotInspireCombinedConfig
    name = "zbot_inspire_combined"

    def __init__(self, config: ZBotInspireCombinedConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connected = False
        
        # Initialize data storage
        self.joint_positions = {}
        self.finger_positions = {}
        self._raw_finger_values = [0] * 6
        self.last_update_time = 0
        
        # Build joint ID mappings
        self.joint_id_to_name = {}
        for i, joint_id in enumerate(config.left_arm_ids):
            if i < len(config.left_arm_names):
                self.joint_id_to_name[joint_id] = config.left_arm_names[i]
        for i, joint_id in enumerate(config.right_arm_ids):
            if i < len(config.right_arm_names):
                self.joint_id_to_name[joint_id] = config.right_arm_names[i]
        
        # Initialize joint positions to zero
        for joint_name in config.left_arm_names + config.right_arm_names:
            self.joint_positions[f"{joint_name}.pos"] = 0.0
            
        # Initialize finger positions to zero
        for finger_name in config.finger_names:
            self.finger_positions[f"{finger_name}.pos"] = 0.0
            
        # OPTIMIZATION: Precompute exponential lookup table for finger conversion
        self._precompute_finger_conversion()

    def _precompute_finger_conversion(self):
        """Precompute finger conversion lookup table for speed."""
        self.finger_lookup = {}
        Xmax = 65535.0
        k = 1.0 / Xmax
        denom = 1.0 - math.exp(-k * Xmax)
        
        # Precompute every possible value (0-65535)
        # This takes ~10ms once during init but saves 3-6ms per packet
        for raw_value in range(0, 65536):
            x = float(raw_value)
            if abs(denom) < 1e-10:
                hand_value = 1000.0 * x / Xmax
            else:
                hand_value = 1000.0 * (1.0 - math.exp(-k * x)) / denom
            
            # Clamp to [0, 1000]
            if hand_value < 0.0:
                hand_value = 0.0
            elif hand_value > 1000.0:
                hand_value = 1000.0
                
            self.finger_lookup[raw_value] = float(hand_value)

    def _convert_udp_to_hand_value(self, raw_value: int) -> float:
        """Fast lookup-based finger conversion (replaces expensive math.exp calls)."""
        # Clamp raw value to valid range
        if raw_value < self.config.raw_min:
            raw_value = self.config.raw_min
        elif raw_value > self.config.raw_max:
            raw_value = self.config.raw_max
            
        # Fast lookup instead of expensive exponential calculation
        return self.finger_lookup.get(raw_value, 0.0)

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        
        # Joint features
        for joint_name in self.config.left_arm_names + self.config.right_arm_names:
            features[f"zbot_{joint_name}.pos"] = float
            
        # Finger features  
        for finger_name in self.config.finger_names:
            features[f"hand_{finger_name}.pos"] = float
            
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
            self.sock.settimeout(self.config.timeout_ms / 1000.0)
            self.connected = True
            logger.info(f"Connected to combined UDP on {self.config.host}:{self.config.port}")
        except Exception as e:
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """UDP-based teleoperator doesn't need calibration."""
        logger.info("Combined UDP teleoperator doesn't require calibration")
        pass

    def configure(self) -> None:
        """UDP-based teleoperator doesn't need configuration."""
        logger.info("Combined UDP teleoperator doesn't require configuration")
        pass

    def setup_motors(self) -> None:
        """UDP-based teleoperator doesn't have motors."""
        logger.info("Combined UDP teleoperator doesn't have motors to setup")
        pass

    def get_action(self) -> dict[str, float]:
        """Receive combined UDP data and return both joint and finger actions."""
        try:
            # Clear buffer to get most recent data (prevent buffer overflow)
            self.sock.setblocking(False)
            latest_data = None
            latest_addr = None
            
            # Read all available packets, keep only the latest
            packet_count = 0
            while True:
                try:
                    data, addr = self.sock.recvfrom(2048)
                    latest_data = data
                    latest_addr = addr
                    packet_count += 1
                except socket.error:
                    break  # No more packets
            
            if latest_data is None:
                # No new data, return last known positions
                action = {}
                action.update({f"zbot_{k}": v for k, v in self.joint_positions.items()})
                action.update({f"hand_{k}": v for k, v in self.finger_positions.items()})
                return action
                
            if packet_count > 1:
                logger.debug(f"Cleared {packet_count-1} old UDP packets, using latest")
                
            message = latest_data.decode('utf-8')
            combined_data = json.loads(message)
            
            # Process joint data
            joints = combined_data.get("joints", {})
            for joint_id_str, position in joints.items():
                joint_id = int(joint_id_str)
                if joint_id in self.joint_id_to_name:
                    joint_name = self.joint_id_to_name[joint_id]
                    joint_key = f"{joint_name}.pos"
                    self.joint_positions[joint_key] = float(position)
            
            # Process finger data
            finger_values = combined_data.get("fingers", [])
            if len(finger_values) >= 6:
                self._raw_finger_values = finger_values[:6]
                
                for i, raw_value in enumerate(finger_values[:6]):
                    if i < len(self.config.finger_names):
                        finger_name = self.config.finger_names[i]
                        hand_value = self._convert_udp_to_hand_value(raw_value)
                        self.finger_positions[f"{finger_name}.pos"] = hand_value
            
            self.last_update_time = time.time()
            
            # Combine all actions with prefixes
            action = {}
            action.update({f"zbot_{k}": v for k, v in self.joint_positions.items()})
            action.update({f"hand_{k}": v for k, v in self.finger_positions.items()})
            
            logger.debug(f"Received combined data: {len(joints)} joints, {len(finger_values)} fingers")
            return action
                
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing combined UDP data: {e}")
            action = {}
            action.update({f"zbot_{k}": v for k, v in self.joint_positions.items()})
            action.update({f"hand_{k}": v for k, v in self.finger_positions.items()})
            return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """UDP teleoperator doesn't send feedback."""
        pass

    def disconnect(self) -> None:
        if self.is_connected:
            self.sock.close()
            self.connected = False
            logger.info("Disconnected from combined UDP teleoperator") 