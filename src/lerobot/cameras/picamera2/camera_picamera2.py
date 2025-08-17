#!/usr/bin/env python3

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_picamera2 import Picamera2CameraConfig

logger = logging.getLogger(__name__)


class Picamera2Camera(Camera):
    """
    Manages camera interactions using picamera2 for Raspberry Pi cameras.

    This class provides a high-level interface to connect to, configure, and read
    frames from Raspberry Pi cameras using picamera2. It follows the RealSense pattern
    with background thread for continuous reading and supports advanced features like
    metadata capture and camera controls.

    Example:
        ```python
        from lerobot.cameras.picamera2 import Picamera2Camera
        from lerobot.cameras.configuration_picamera2 import Picamera2CameraConfig

        # Basic usage
        config = Picamera2CameraConfig(camera_index=0)
        camera = Picamera2Camera(config)
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

    def __init__(self, config: Picamera2CameraConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._frame_count = 0
        
        # picamera2 objects
        self._picam2 = None
        self._latest_metadata = None
        
        # Background thread for async reading
        self.thread = None
        self.stop_event = None
        self.frame_lock = Lock()
        self.latest_frame = None
        self.new_frame_event = Event()
        
        # Warmup settings
        self.warmup_s = 1.0

    def __str__(self) -> str:
        return f"Picamera2Camera(index={self.config.camera_index})"

    @property
    def is_connected(self) -> bool:
        """Check if the camera is currently connected."""
        return self._connected and self._picam2 is not None

    def _apply_hardcoded_crop(self, frame: np.ndarray) -> np.ndarray:
        """Hardcoded crop: remove 420px from left and right, output 1080x1080."""
        # Input: 1200*1600, Output: 400x500
        # Remove 420px from left and right (1920 - 840 = 1080)
        return frame[650:1050, 700:1200]  # Crop from x=420 to x=1500

    def connect(self, warmup: bool = True) -> None:
        """Connect to the camera using picamera2."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        try:
            from picamera2 import Picamera2
            
            # Create and configure picamera2
            self._picam2 = Picamera2(camera_num=self.config.camera_index)
            logger.info(f"Created Picamera2 instance: {self._picam2}")
            
            # Get camera info
            camera_info = self._picam2.camera_config
            logger.info(f"Camera config: {camera_info}")
            
            # HARDCODED: Always capture at 1080p regardless of config
            config = self._picam2.create_video_configuration(
                main={
                    "format": 'RGB888',  # Use RGB888 for correct colors (was XRGB8888)
                    "size": (1600, 1200)  # HARDCODED 1080p
                },
                controls={
                    "FrameDurationLimits": (33333, 33333),  # 30 FPS
                    "AeEnable": True,  # Auto exposure
                    "AwbEnable": True,  # Auto white balance (revert to auto)
                    "NoiseReductionMode": 1,  # Enable noise reduction
                }
            )
            
            self._picam2.configure(config)
            logger.info("Camera HARDCODED to 1920x1080 (ignoring config resolution)")
            
            # Start the camera
            self._picam2.start()
            logger.info("Camera started")
            
            # Mark as connected before warmup
            self._connected = True
            
            if warmup:
                time.sleep(self.warmup_s)
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    try:
                        # Use direct picamera2 call for warmup
                        frame = self._picam2.capture_array()
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Warmup frame capture failed: {e}")
                        break

            logger.info(f"{self} connected.")

        except ImportError:
            raise ImportError("picamera2 is required. Install with: sudo apt install python3-picamera2")
        except Exception as e:
            logger.error(f"Failed to connect {self}: {e}")
            self._cleanup()
            raise

    def _cleanup(self):
        """Clean up picamera2 resources."""
        if self._picam2 is not None:
            try:
                self._picam2.close()
            except:
                pass
            self._picam2 = None
        self._connected = False

    def disconnect(self) -> None:
        """Disconnect from the camera and clean up resources."""
        if not self._connected:
            return

        # Stop background thread
        if self.thread is not None:
            self._stop_read_thread()

        # Clean up picamera2 resources
        self._cleanup()
        logger.info(f"{self} disconnected.")

    def set_controls(self, controls: dict) -> None:
        """
        Set camera controls (exposure, gain, etc.).
        
        Args:
            controls: Dictionary of camera controls to set
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            self._picam2.set_controls(controls)
            logger.info(f"Set camera controls: {controls}")
        except Exception as e:
            logger.error(f"Failed to set camera controls: {e}")
            raise

    def get_metadata(self) -> dict:
        """
        Get the latest camera metadata.
        
        Returns:
            dict: Camera metadata including exposure, gain, timestamp, etc.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            metadata = self._picam2.capture_metadata()
            self._latest_metadata = metadata
            return metadata
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return {}

    def read_fast(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Fast non-blocking read that doesn't wait for camera hardware.
        Returns the latest available frame or raises TimeoutError.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # Capture frame as numpy array (like OpenCV)
            frame = self._picam2.capture_array()
            
            # Apply hardcoded crop: 1920x1080 -> 1080x1080 (remove 420px from left/right)
            frame = self._apply_hardcoded_crop(frame)
            
            # Frame is already in RGB888 format - convert to BGR if needed
            target_color_mode = color_mode if color_mode is not None else self.config.color_mode
            if target_color_mode == ColorMode.BGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # If target is RGB, frame is already in correct format
            
            self._frame_count += 1
            return frame
            
        except Exception as e:
            raise TimeoutError(f"Fast read failed: {e}")

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single frame synchronously from the camera.

        This is a blocking call. It waits for a frame from the camera hardware.

        Args:
            color_mode: Target color mode (RGB or BGR). If None, uses config default.
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The captured frame as a NumPy array (height, width, channels).

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames fails.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # Capture frame as numpy array (like OpenCV)
            frame = self._picam2.capture_array()
            
            # Apply hardcoded crop: 1920x1080 -> 1080x1080 (remove 420px from left/right)
            frame = self._apply_hardcoded_crop(frame)
            
            # Frame is already in RGB888 format - convert to BGR if needed
            target_color_mode = color_mode if color_mode is not None else self.config.color_mode
            if target_color_mode == ColorMode.BGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # If target is RGB, frame is already in correct format
            
            # Skip metadata capture for speed - it's not essential for robot control
            # try:
            #     metadata = self._picam2.capture_metadata()
            #     self._latest_metadata = metadata
            # except:
            #     pass  # Metadata capture is optional
            
            self._frame_count += 1
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            raise

    def _read_loop(self):
        """Internal loop run by the background thread for asynchronous reading."""
        while not self.stop_event.is_set():
            try:
                # Use fast non-blocking read for maximum speed
                frame = self.read_fast()
                
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

            except (TimeoutError, DeviceNotConnectedError):
                # Small delay when no frame available or disconnected
                time.sleep(0.001)
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")
                # Small delay to prevent tight loop on errors
                time.sleep(0.001)

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 50) -> np.ndarray:
        """
        Reads the latest available frame data asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 50ms for faster response.

        Returns:
            np.ndarray: The latest captured frame data, processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # Use shorter timeout for faster response
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @property
    def get_frame_count(self) -> int:
        """Get the number of frames captured."""
        return self._frame_count

    @property
    def latest_metadata(self) -> dict:
        """Get the latest camera metadata."""
        return self._latest_metadata or {}

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Find available picamera2 cameras.
        
        Returns:
            list: List of camera information dictionaries
        """
        cameras = []
        try:
            from picamera2 import Picamera2
            
            # Try to create a Picamera2 instance to check availability
            # This is more reliable than global_camera_info() which has numpy compatibility issues
            picam2 = Picamera2()
            camera_info = {
                "name": "picamera2-0",
                "type": "picamera2",
                "id": "0",
                "index": 0,
                "model": "Raspberry Pi Camera"
            }
            cameras.append(camera_info)
            picam2.close()
            
        except Exception as e:
            # Log the error but don't fail completely - just return empty list
            logger.debug(f"Error finding picamera2 cameras: {e}")
            # Try a simpler approach - just return a default camera
            cameras = [{
                "name": "picamera2-0",
                "type": "picamera2", 
                "id": "0",
                "index": 0,
                "model": "Raspberry Pi Camera"
            }]

        return cameras 