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

"""
Provides the LibCameraCamera class for capturing frames from Raspberry Pi cameras using libcamera.
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_libcamera import LibCameraCameraConfig

logger = logging.getLogger(__name__)


class LibCameraCamera(Camera):
    """
    Manages camera interactions using libcamera for Raspberry Pi cameras.

    This class provides a high-level interface to connect to, configure, and read
    frames from Raspberry Pi cameras using libcamera-still.

    Example:
        ```python
        from lerobot.cameras.libcamera import LibCameraCamera
        from lerobot.cameras.configuration_libcamera import LibCameraCameraConfig

        # Basic usage
        config = LibCameraCameraConfig(camera_index=0)
        camera = LibCameraCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera
        camera.disconnect()
        ```
    """

    def __init__(self, config: LibCameraCameraConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._temp_dir = None
        self._frame_count = 0

    def __str__(self) -> str:
        return f"LibCameraCamera(index={self.config.camera_index})"

    @property
    def is_connected(self) -> bool:
        """Check if the camera is currently connected."""
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        """Connect to the camera using libcamera-still."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        try:
            # Test if libcamera-still is available
            result = subprocess.run(
                ["libcamera-still", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("libcamera-still not available")

            # Create temporary directory for images
            self._temp_dir = tempfile.mkdtemp(prefix="libcamera_")
            self._connected = True
            logger.info(f"{self} connected")

        except Exception as e:
            logger.error(f"Failed to connect to {self}: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        if not self._connected:
            return

        self._connected = False
        if self._temp_dir:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        logger.info(f"{self} disconnected")

    def read(self) -> np.ndarray:
        """Capture and return a single frame from the camera."""
        if not self._connected:
            raise DeviceNotConnectedError("Camera not connected")
            
        try:
            # Use libcamera-vid for video-like capture (much faster than libcamera-still)
            cmd = [
                "libcamera-vid",
                "--nopreview",
                "--timeout", "1000",
                "--width", str(self.config.width),
                "--height", str(self.config.height),
                "--framerate", str(self.config.fps),
                "--output", "-",  # Output to stdout instead of file
                "--frames", "1"    # Capture only 1 frame
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=1.0  # 1 second timeout
            )
            
            if result.returncode != 0:
                # Fallback to libcamera-still if vid fails
                cmd_still = [
                    "libcamera-still",
                    "--nopreview",
                    "--timeout", "1000",
                    "--width", str(self.config.width),
                    "--height", str(self.config.height),
                    "--quality", str(self.config.quality),
                    "-o", f"{self._temp_dir}/temp.jpg"
                ]
                
                result = subprocess.run(
                    cmd_still,
                    capture_output=True,
                    text=True,
                    timeout=2.0
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"libcamera capture failed: {result.stderr}")
                    
                # Read the captured image with OpenCV
                image_path = f"{self._temp_dir}/temp.jpg"
                frame = cv2.imread(image_path)
                if frame is None:
                    raise RuntimeError(f"Failed to read captured image: {image_path}")
            else:
                # Parse raw video data from stdout
                # libcamera-vid outputs raw video frames to stdout
                raw_data = result.stdout
                if len(raw_data) == 0:
                    raise RuntimeError("No data captured from libcamera-vid")
                
                # Convert raw data to numpy array
                # Assuming RGB format from libcamera-vid
                frame = np.frombuffer(raw_data, dtype=np.uint8)
                frame = frame.reshape((self.config.height, self.config.width, 3))
                
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Convert BGR to RGB (like OpenCV camera)
            # Default to BGR since config doesn't have color_mode
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self._frame_count += 1
            return frame
            
        except subprocess.TimeoutExpired:
            raise TimeoutError("Camera capture timed out")
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            raise

    def async_read(self, timeout_ms: float = 1000.0) -> np.ndarray:
        """Asynchronously capture and return a single frame from the camera."""
        # For libcamera, async_read is the same as read since libcamera-still is blocking
        return self.read()

    @property
    def get_frame_count(self) -> int:
        """Get the number of frames captured."""
        return self._frame_count

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Find available libcamera cameras.
        
        Returns:
            list: List of camera information dictionaries
        """
        cameras = []
        
        try:
            # Check if libcamera-still is available
            result = subprocess.run(
                ["libcamera-still", "--list-cameras"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse camera list
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line and '[' in line:
                        # Parse camera info
                        parts = line.split(':')
                        if len(parts) >= 2:
                            camera_id = parts[0].strip()
                            camera_info = parts[1].strip()
                            
                            # Only add if it looks like a real camera (has numeric ID)
                            if camera_id.isdigit():
                                cameras.append({
                                    'id': camera_id,
                                    'name': f"LibCamera {camera_id}",
                                    'info': camera_info,
                                    'type': 'libcamera'
                                })
            
        except Exception as e:
            logger.warning(f"Error finding libcamera cameras: {e}")
        
        return cameras 