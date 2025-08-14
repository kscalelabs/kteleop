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
        # Combine zbot motor obs, hand obs, and cameras
        zbot_obs = {f"zbot_{k}": v for k, v in self.zbot.observation_features.items() if isinstance(v, type)}
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
        z = self.zbot.get_observation()
        obs.update({f"zbot_{k}": v for k, v in z.items()})
        
        # Get Inspire hand observation  
        hand_start = time.perf_counter()
        h = self.hand.get_observation()
        obs.update({f"hand_{k}": v for k, v in h.items()})
        hand_time = (time.perf_counter() - hand_start) * 1000

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs

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