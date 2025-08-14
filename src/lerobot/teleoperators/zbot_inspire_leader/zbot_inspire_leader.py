#!/usr/bin/env python

import logging
from functools import cached_property

from lerobot.teleoperators.oymotion_glove.config_oymotion_glove import OyMotionGloveConfig
from lerobot.teleoperators.oymotion_glove.oymotion_glove import OyMotionGlove
from lerobot.teleoperators.zbot_leader.config_zbot_leader import ZbotLeaderConfig
from lerobot.teleoperators.zbot_leader.zbot_leader import ZbotLeader

from ..teleoperator import Teleoperator
from .config_zbot_inspire_leader import ZBotInspireLeaderConfig

logger = logging.getLogger(__name__)


class ZBotInspireLeader(Teleoperator):
    """Combined teleop: ZBot leader + OyMotion glove.
    Action keys are prefixed: zbot_..., hand_...
    """

    config_class = ZBotInspireLeaderConfig
    name = "zbot_inspire_leader"

    def __init__(self, config: ZBotInspireLeaderConfig):
        super().__init__(config)
        self.config = config

        zbot_cfg = ZbotLeaderConfig(
            id=f"{config.id}_zbot" if config.id else None,
            host=config.zbot_host,
            port=config.zbot_port,
        )
        glove_cfg = OyMotionGloveConfig(
            id=f"{config.id}_glove" if config.id else None,
            host=config.glove_host,
            port=config.glove_port,
        )

        self.zbot = ZbotLeader(zbot_cfg)
        self.glove = OyMotionGlove(glove_cfg)

    @cached_property
    def action_features(self) -> dict[str, type]:
        zbot_ft = {f"zbot_{k}": v for k, v in self.zbot.action_features.items()}
        glove_ft = {f"hand_{k}": v for k, v in self.glove.action_features.items()}
        return {**zbot_ft, **glove_ft}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.zbot.is_connected and self.glove.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.zbot.connect(calibrate)
        self.glove.connect(calibrate)
        # Make both receivers non-blocking to avoid double-blocking lag
        try:
            if hasattr(self.zbot, "sock") and self.zbot.sock:
                self.zbot.sock.setblocking(False)
        except Exception:
            pass
        try:
            if hasattr(self.glove, "sock") and self.glove.sock:
                self.glove.sock.settimeout(0.0)
        except Exception:
            pass

    @property
    def is_calibrated(self) -> bool:
        return self.zbot.is_calibrated and self.glove.is_calibrated

    def calibrate(self) -> None:
        self.zbot.calibrate()
        self.glove.calibrate()

    def configure(self) -> None:
        self.zbot.configure()
        self.glove.configure()

    def setup_motors(self) -> None:
        self.zbot.setup_motors()
        self.glove.setup_motors()

    def get_action(self) -> dict[str, float]:
        zbot_action = self.zbot.get_action()
        glove_action = self.glove.get_action()
        out = {**{f"zbot_{k}": v for k, v in zbot_action.items()}, **{f"hand_{k}": v for k, v in glove_action.items()}}
        return out

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Split and forward if needed
        zbot_fb = {k.removeprefix("zbot_"): v for k, v in feedback.items() if k.startswith("zbot_")}
        glove_fb = {k.removeprefix("hand_"): v for k, v in feedback.items() if k.startswith("hand_")}
        if zbot_fb:
            self.zbot.send_feedback(zbot_fb)
        if glove_fb:
            self.glove.send_feedback(glove_fb)

    def disconnect(self) -> None:
        self.zbot.disconnect()
        self.glove.disconnect() 