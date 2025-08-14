#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.teleoperator import TeleoperatorConfig


@dataclass
class ZBotInspireCombinedConfig(TeleoperatorConfig):
    """
    Configuration for combined ZBot + Inspire hand teleoperator over single UDP port.
    Receives both joint and finger data in one packet.
    """
    host: str = "0.0.0.0"
    port: int = 8888
    timeout_ms: int = 100
    
    # Joint configuration (inherited from ZBot)
    left_arm_ids: list[int] = None
    right_arm_ids: list[int] = None
    left_arm_names: list[str] = None
    right_arm_names: list[str] = None
    
    # Finger configuration (inherited from Inspire)
    finger_names: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky", "extra")
    raw_min: int = 0
    raw_max: int = 65535

    def __post_init__(self):
        if self.left_arm_ids is None:
            self.left_arm_ids = [11, 12, 13, 14, 15]
        if self.right_arm_ids is None:
            self.right_arm_ids = [21, 22, 23, 24, 25]
        if self.left_arm_names is None:
            self.left_arm_names = [
                "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                "left_elbow", "left_wrist"
            ]
        if self.right_arm_names is None:
            self.right_arm_names = [
                "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", 
                "right_elbow", "right_wrist"
            ]


TeleoperatorConfig.register_subclass("zbot_inspire_combined", ZBotInspireCombinedConfig) 