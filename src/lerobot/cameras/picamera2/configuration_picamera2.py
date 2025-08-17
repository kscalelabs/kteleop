#!/usr/bin/env python3

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode


@dataclass(kw_only=True)
class Picamera2CameraConfig(CameraConfig):
    """Configuration for picamera2 camera."""
    
    type: str = "picamera2"
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    color_mode: ColorMode = ColorMode.RGB




# Register with CameraConfig
CameraConfig.register_subclass("picamera2")(Picamera2CameraConfig) 