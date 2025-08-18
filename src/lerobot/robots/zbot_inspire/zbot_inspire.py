#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.inspire_hand import InspireHand, InspireHandConfig
from lerobot.robots.zbot import ZBot
from lerobot.robots.zbot.config_zbot import ZBotConfig

from ..robot import Robot
from .config_zbot_inspire import ZBotInspireConfig

logger = logging.getLogger(__name__)


class ZBotInspire(Robot):
    """
    Combined robot: ZBot (upper body) + Inspire hand.
    Action/observation keys are prefixed: `zbot_...` and `hand_...`.
    Cameras belong to the combo robot (if provided).
    """

    config_class = ZBotInspireConfig
    name = "zbot_inspire"

    def __init__(self, config: ZBotInspireConfig):
        super().__init__(config)
        self.config = config

        # Build sub-robots with mapped configs
        zbot_cfg = ZBotConfig(
            id=f"{config.id}_zbot" if config.id else None,
            calibration_dir=config.calibration_dir,
            left_arm_ids=config.left_arm_ids,
            right_arm_ids=config.right_arm_ids,
            left_arm_names=config.left_arm_names,
            right_arm_names=config.right_arm_names,
            cameras={},  # cameras managed by combo
        )
        self.zbot = ZBot(zbot_cfg)

        hand_cfg = InspireHandConfig(
            id=f"{config.id}_hand" if config.id else None,
            serial_port=config.serial_port,
            baudrate=config.baudrate,
            hand_id=config.hand_id,
            finger_names=(config.finger_names if config.finger_names is not None else None),
        )
        self.hand = InspireHand(hand_cfg)

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _zbot_ft(self) -> dict[str, type]:
        return {f"zbot_{k}": v for k, v in self.zbot.action_features.items()} | {
            f"zbot_{k}": v for k, v in self.zbot.observation_features.items() if isinstance(v, type)
        }

    @property
    def _hand_ft(self) -> dict[str, type]:
        return {f"hand_{k}": v for k, v in self.hand.action_features.items()} | {
            f"hand_{k}": v for k, v in self.hand.observation_features.items() if isinstance(v, type)
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Combine zbot motor obs (POSITION ONLY), hand obs, and cameras
        zbot_obs = {f"zbot_{k}": v for k, v in self.zbot.observation_features.items() 
                    if isinstance(v, type) and k.endswith('.pos')}  # Only position, no velocity
        hand_obs = {f"hand_{k}": v for k, v in self.hand.observation_features.items() if isinstance(v, type)}
        return {**zbot_obs, **hand_obs, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        zbot_act = {f"zbot_{k}": v for k, v in self.zbot.action_features.items()}
        hand_act = {f"hand_{k}": v for k, v in self.hand.action_features.items()}
        return {**zbot_act, **hand_act}

    @property
    def is_connected(self) -> bool:
        return (
            self.zbot.is_connected and self.hand.is_connected and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.zbot.connect(calibrate)
        self.hand.connect(calibrate)
        
        # Set hand initial positions after connection
        if calibrate:
            self._set_hand_initial_positions()
            
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.zbot.is_calibrated and self.hand.is_calibrated

    def calibrate(self) -> None:
        self.zbot.calibrate()
        self.hand.calibrate()

    def configure(self) -> None:
        self.zbot.configure()
        self.hand.configure()

    def setup_motors(self) -> None:
        self.zbot.setup_motors()
        self.hand.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs = {}
        
        # Get ZBot observation with timeout protection
        try:
            zbot_start = time.perf_counter()
            z = self.zbot.get_observation()
            # Only keep position observations, filter out velocity
            obs.update({f"zbot_{k}": v for k, v in z.items() if k.endswith('.pos')})
            zbot_time = (time.perf_counter() - zbot_start) * 1000
            if zbot_time > 100:  # Warn if ZBot takes >100ms
                logger.warning(f"ZBot observation took {zbot_time:.1f}ms")
        except Exception as e:
            logger.error(f"ZBot observation failed: {e}")
            # Return dummy ZBot data to keep going (position only)
            for joint_name in ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
                             "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist"]:
                obs[f"zbot_{joint_name}.pos"] = 0.0
                # Velocity removed - only position needed for pick & place tasks
        
        # Get Inspire hand observation with timeout protection
        try:
            hand_start = time.perf_counter()
            h = self.hand.get_observation()
            obs.update({f"hand_{k}": v for k, v in h.items()})
            hand_time = (time.perf_counter() - hand_start) * 1000
            if hand_time > 100:  # Warn if hand takes >100ms
                logger.warning(f"Hand observation took {hand_time:.1f}ms")
        except Exception as e:
            logger.error(f"Hand observation failed: {e}")
            # Return dummy hand data to keep going
            for finger_name in ["thumb", "index", "middle", "ring", "pinky", "extra"]:
                obs[f"hand_{finger_name}.pos"] = 0.0

        # Get camera observations with timeout protection
        for cam_key, cam in self.cameras.items():
            try:
                start = time.perf_counter()
                obs[cam_key] = cam.async_read(timeout_ms=100)  # 100ms timeout for cameras
                dt_ms = (time.perf_counter() - start) * 1e3
                #if dt_ms > 50:  # Warn if camera takes >50ms
                    #logger.warning(f"Camera {cam_key} took {dt_ms:.1f}ms")
            except Exception as e:
                logger.warning(f"Camera {cam_key} read failed: {e}")
                # Return dummy camera data to keep dataset happy
                # Create a black image with the expected dimensions
                try:
                    height, width, _ = self._cameras_ft[cam_key]
                    import numpy as np
                    obs[cam_key] = np.zeros((height, width, 3), dtype=np.uint8)
                except:
                    # Fallback to a small black image if dimensions unknown
                    obs[cam_key] = np.zeros((480, 640, 3), dtype=np.uint8)

        return obs

    def _set_hand_initial_positions(self) -> None:
        """Set the Inspire hand to initial positions."""
        if not self.hand.is_connected:
            logger.warning("Hand not connected, skipping initial position setting")
            return
            
        logger.info("Setting Inspire hand to initial positions...")
        
        # Initial positions for all 6 fingers (in hand units 0-1000)
        # Based on the values you provided: 545.2890014648438, 362.95208740234375, 327.1883850097656, 492.9804992675781, 486.6811218261719, 70.99323272705078
        initial_positions = {
            "pinky": 545.2890014648438,      # Finger 0
            "ring": 362.95208740234375,      # Finger 1  
            "middle": 327.1883850097656,     # Finger 2
            "index": 492.9804992675781,      # Finger 3
            "thumb": 486.6811218261719,      # Finger 4
            "extra": 70.99323272705078       # Finger 5
        }
        
        try:
            # Move all fingers to initial positions
            for finger_name, target_pos in initial_positions.items():
                logger.info(f"Moving {finger_name} to {target_pos:.2f}")
                
                # Send position command to hand
                self.hand.send_action({f"{finger_name}.pos": target_pos / 1000.0})  # Convert to normalized 0-1
                
            logger.info("Inspire hand initial positions set successfully")
            
        except Exception as e:
            logger.error(f"Failed to set hand initial positions: {e}")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        zbot_action = {k.removeprefix("zbot_"): v for k, v in action.items() if k.startswith("zbot_")}
        hand_action = {k.removeprefix("hand_"): v for k, v in action.items() if k.startswith("hand_")}
        sent_z = self.zbot.send_action(zbot_action)
        sent_h = self.hand.send_action(hand_action)
        return {**{f"zbot_{k}": v for k, v in sent_z.items()}, **{f"hand_{k}": v for k, v in sent_h.items()}}

    def disconnect(self) -> None:
        self.zbot.disconnect()
        self.hand.disconnect()
        for cam in self.cameras.values():
            cam.disconnect() 