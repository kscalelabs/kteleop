#!/usr/bin/env python3

import logging
import numpy as np
import cv2
from typing import Any

from ..camera import Camera
from ..configs import ColorMode
from .configuration_picamera2 import Picamera2CameraConfig

logger = logging.getLogger(__name__)


class Picamera2Camera(Camera):
    """
    Raspberry Pi Camera using picamera2 library.
    
    This provides a much more OpenCV-like interface compared to libcamera command-line tools.
    """

    def __init__(self, config: Picamera2CameraConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._picam2 = None
        self._frame_count = 0

    def __str__(self) -> str:
        return f"Picamera2Camera(index={self.config.camera_index})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._picam2 is not None

    def connect(self, warmup: bool = True) -> None:
        """Connect to the camera using picamera2."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from picamera2 import Picamera2
            
            # Create and configure picamera2
            self._picam2 = Picamera2()
            
            # Configure for the specified resolution
            config = self._picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (self.config.width, self.config.height)}
            )
            self._picam2.configure(config)
            
            # Start the camera
            self._picam2.start()
            
            self._connected = True
            logger.info(f"{self} connected")
            
        except ImportError:
            raise ImportError("picamera2 is required. Install with: pip install picamera2")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to camera: {e}")

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        if not self._connected:
            return

        if self._picam2 is not None:
            self._picam2.close()
            self._picam2 = None
            
        self._connected = False
        logger.info(f"{self} disconnected")

    def read(self) -> np.ndarray:
        """Capture and return a single frame from the camera."""
        if not self._connected:
            raise DeviceNotConnectedError("Camera not connected")
            
        try:
            # Capture frame as numpy array (like OpenCV)
            frame = self._picam2.capture_array()
            
            # Convert XRGB to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Convert BGR to RGB if needed
            if self.config.color_mode != ColorMode.BGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self._frame_count += 1
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            raise

    def async_read(self, timeout_ms: float = 1000.0) -> np.ndarray:
        """Asynchronously capture and return a single frame from the camera."""
        # For picamera2, async_read is the same as read since capture_array is fast
        return self.read()

    @property
    def get_frame_count(self) -> int:
        return self._frame_count

    @staticmethod
    def find_cameras() -> list[int]:
        """Find available picamera2 cameras."""
        try:
            from picamera2 import Picamera2
            
            # picamera2 typically only supports one camera at a time
            # Return [0] if we can create a Picamera2 instance
            picam2 = Picamera2()
            picam2.close()
            return [0]
        except Exception:
            return [] 