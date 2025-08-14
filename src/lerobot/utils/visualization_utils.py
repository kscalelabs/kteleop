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

import os
import time
from typing import Any

import numpy as np
import rerun as rr


def _init_rerun(session_name: str = "lerobot_control_loop", web_mode: bool = False) -> None:
    """Initializes the Rerun SDK for visualizing the control loop.
    
    Args:
        session_name: Name for the Rerun session
        web_mode: If True, serve web viewer instead of spawning native GUI
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    
    if web_mode:
        # Start web viewer for headless/remote systems
        try:
            rr.serve_web(
                open_browser=False,  # Don't try to open browser on headless system
                web_port=9091,       # Changed from 9090 to avoid conflicts
                ws_port=9878,        # Changed from 9877 to avoid conflicts
                server_memory_limit=memory_limit
            )
            print(f"🌐 Rerun web viewer started on port 9091")
            print(f"📡 WebSocket server running on port 9878")
            print(f"")
            print(f"🖥️  To access from your Mac:")
            print(f"   1. SSH with port forwarding: ssh -L 9091:localhost:9091 dpsh@kbotv2-no8")
            print(f"   2. Open browser to: http://localhost:9091")
            print(f"")
            print(f"🌍 Alternatively, if on same network:")
            print(f"   - Try: http://kbotv2-no8:9091")
            print(f"   - Or check Pi IP with 'hostname -I' and use: http://PI_IP:9091")
        except Exception as e:
            print(f"❌ Failed to start web viewer: {e}")
            print("💡 Falling back to native spawn...")
            rr.spawn(memory_limit=memory_limit)
    else:
        # Use native GUI (original behavior)
        rr.spawn(memory_limit=memory_limit)


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    total_start = time.perf_counter()
    log_count = 0
    
    # Log observations
    obs_start = time.perf_counter()
    for obs, val in observation.items():
        log_start = time.perf_counter()
        if isinstance(val, float):
            rr.log(f"observation.{obs}", rr.Scalar(val))
            log_count += 1
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"observation.{obs}_{i}", rr.Scalar(float(v)))
                    log_count += 1
            else:
                rr.log(f"observation.{obs}", rr.Image(val), static=True)
                log_count += 1
        log_time = (time.perf_counter() - log_start) * 1000
        if log_time > 10:  # Only print if > 10ms
            print(f"⏱️  Slow log {obs}: {log_time:.1f}ms")
    
    obs_time = (time.perf_counter() - obs_start) * 1000
    
    # Log actions  
    act_start = time.perf_counter()
    for act, val in action.items():
        log_start = time.perf_counter()
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalar(val))
            log_count += 1
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}_{i}", rr.Scalar(float(v)))
