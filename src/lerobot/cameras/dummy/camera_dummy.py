#!/usr/bin/env python

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

import logging
import time
from typing import Any

import numpy as np
import cv2

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from .configuration_dummy import DummyCameraConfig

logger = logging.getLogger(__name__)


class DummyCamera(Camera):
    """
    Dummy camera that returns valid data as fast as possible.
    Used for testing camera framework performance without hardware overhead.
    """

    config_class = DummyCameraConfig
    name = "dummy"

    def __init__(self, config: DummyCameraConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._frame_count = 0
        self._latest_frame = None
        self._latest_metadata = {}

    def __str__(self) -> str:
        return f"DummyCamera(index={self.config.camera_index})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        """Connect to the dummy camera - instant connection."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        # Create a dummy frame immediately
        self._latest_frame = self._generate_dummy_frame()
        self._connected = True
        logger.info(f"{self} connected instantly.")

    def _generate_dummy_frame(self) -> np.ndarray:
        """Generate a dummy frame as fast as possible."""
        # Create a simple colored frame with the configured dimensions
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        
        # Add some simple pattern for visual identification
        frame[:, :, 0] = 128  # Red channel
        frame[:, :, 1] = 64   # Green channel  
        frame[:, :, 2] = 192  # Blue channel
        
        # Add frame counter in corner
        cv2.putText(frame, f"Frame {self._frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def disconnect(self) -> None:
        """Disconnect from the dummy camera."""
        if not self._connected:
            return

        self._connected = False
        self._latest_frame = None
        logger.info(f"{self} disconnected.")

    def set_controls(self, controls: dict) -> None:
        """Set camera controls - dummy implementation."""
        logger.debug(f"Setting dummy camera controls: {controls}")

    def get_metadata(self) -> dict:
        """Get camera metadata - dummy implementation."""
        return {
            "frame_count": self._frame_count,
            "timestamp": time.time(),
            "dummy": True
        }

    def read(self, color_mode: Any = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Read a frame from the dummy camera - instant return.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Generate new frame and return immediately
        self._frame_count += 1
        frame = self._generate_dummy_frame()
        self._latest_frame = frame
        
        return frame

    def async_read(self, timeout_ms: float = 50) -> np.ndarray:
        """
        Read frame asynchronously - instant return with no waiting.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Return latest frame immediately - no background thread needed
        self._frame_count += 1
        frame = self._generate_dummy_frame()
        self._latest_frame = frame
        
        return frame

    @property
    def get_frame_count(self) -> int:
        """Get the number of frames captured."""
        return self._frame_count

    @property
    def latest_metadata(self) -> dict:
        """Get the latest camera metadata."""
        return self._latest_metadata

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Find available dummy cameras.
        
        Returns:
            list: List of camera information dictionaries
        """
        return [{
            "name": "dummy-0",
            "type": "dummy",
            "id": "0",
            "index": 0,
            "model": "Dummy Camera"
        }] 