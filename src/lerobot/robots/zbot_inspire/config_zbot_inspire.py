#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import List

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("zbot_inspire")
@dataclass
class ZBotInspireConfig(RobotConfig):
    """
    Combined robot: ZBot upper body + Inspire hand.
    Cameras (if any) are owned by this combo robot (not the sub-robots).
    """

    # ZBot joint configuration (same defaults as ZBotConfig)
    left_arm_ids: List[int] = field(default_factory=lambda: [11, 12, 13, 14, 15])
    right_arm_ids: List[int] = field(default_factory=lambda: [21, 22, 23, 24, 25])

    left_arm_names: List[str] = field(default_factory=lambda: [
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist",
    ])
    right_arm_names: List[str] = field(default_factory=lambda: [
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist",
    ])

    # Inspire hand serial configuration (same semantics as InspireHandConfig)
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    hand_id: int = 1

    # Optional finger names (if None, Inspire defaults apply)
    finger_names: list[str] | None = None

    # Cameras for the combined robot
    cameras: dict[str, CameraConfig] = field(default_factory=dict) 