#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("zbot_inspire_leader")
@dataclass
class ZBotInspireLeaderConfig(TeleoperatorConfig):
    # ZBot leader UDP
    zbot_host: str = "0.0.0.0"
    zbot_port: int = 8888

    # OyMotion glove UDP
    glove_host: str = "10.33.10.154"
    glove_port: int = 8889 