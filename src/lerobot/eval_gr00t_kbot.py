# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KBot (ZBot Inspire) Real Robot evaluation script for GR00T policies.

Example command:

```shell
python eval_gr00t_kbot.py     --use_policy     --host=10.33.11.7     --port=5555     --lang_instruction="Pick up the bottle and place it in the container."     --serial_port=/dev/ttyUSB0
```

First test the robot setup:
```shell
python eval_gr00t_kbot.py \
    --dataset_path=./demo_data/pick_bottle/ \
    --cam_left_idx=0 \
    --cam_right_idx=1 \
    --serial_port=/dev/ttyUSB0
```
"""

import time
from contextlib import contextmanager
import sys
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.robots.zbot_inspire import ZBotInspire, ZBotInspireConfig
from lerobot.cameras.picamera2.configuration_picamera2 import Picamera2CameraConfig

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################


class KBotRobot:
    def __init__(self, calibrate=False, enable_cameras=True, cam_left_idx=0, cam_right_idx=1, serial_port="/dev/ttyUSB0"):
        self.calibrate_on_connect = calibrate
        self.enable_cameras = enable_cameras
        self.cam_left_idx = cam_left_idx
        self.cam_right_idx = cam_right_idx
        self.serial_port = serial_port
        
        # Create camera configs if enabled
        cameras = {}
        if enable_cameras:
            cameras = {
                "left": Picamera2CameraConfig(
                    camera_index=cam_left_idx,
                    width=500,
                    height=400,
                    fps=30
                ),
                "right": Picamera2CameraConfig(
                    camera_index=cam_right_idx,
                    width=500,
                    height=400,
                    fps=30
                )
            }
        
        # Create ZBotInspire configuration
        self.config = ZBotInspireConfig(
            id="kbot",
            serial_port=serial_port,
            baudrate=115200,
            hand_id=1,
            cameras=cameras,
            finger_names=["pinky", "ring", "middle", "index", "thumb", "extra"]
        )
        
        # Create the robot
        self.robot = ZBotInspire(self.config)

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        print("================> Connecting KBot Robot =================")
        self.robot.connect(calibrate=self.calibrate_on_connect)
        print("================> KBot Robot is fully connected =================")

    def move_to_initial_pose(self):
        print("-------------------------------- moving to initial pose")
        # Define a safe initial pose for the dual-arm robot
        # Angles in degrees - adjust these based on your robot's safe configuration
        initial_state = {
            "zbot_left_shoulder_pitch.pos": 2.260669469833374,
            "zbot_left_shoulder_roll.pos": 1.071106195449829,
            "zbot_left_shoulder_yaw.pos": 33.900001525878906,
            "zbot_left_elbow.pos": 2.1392993927001953,
            "zbot_left_wrist.pos": -59.19999694824219,
            "zbot_right_shoulder_pitch.pos": -10.482698440551758,
            "zbot_right_shoulder_roll.pos": -5.94090461730957,
            "zbot_right_shoulder_yaw.pos": -2.7355010509490967,
            "zbot_right_elbow.pos": 84.92172241210938,
            "zbot_right_wrist.pos": 20.519702911376953,
            "hand_pinky.pos": 267.0544738769531,
            "hand_ring.pos": 259.8277282714844,
            "hand_middle.pos": 278.8610534667969,
            "hand_index.pos": 366.0862731933594,
            "hand_thumb.pos": 325.3300476074219,
            "hand_extra.pos": 410.1182861328125
        }
        
        # Send initial state to robot
        self.robot.send_action(initial_state)
        time.sleep(2)

    def go_home(self):
        print("-------------------------------- moving to home pose")
        # Define home pose - safe resting position
        home_state = {
            "zbot_left_shoulder_pitch.pos": 0.0,
            "zbot_left_shoulder_roll.pos": 45.0,
            "zbot_left_shoulder_yaw.pos": 0.0,
            "zbot_left_elbow.pos": -90.0,
            "zbot_left_wrist.pos": 0.0,
            "zbot_right_shoulder_pitch.pos": 0.0,
            "zbot_right_shoulder_roll.pos": -45.0,
            "zbot_right_shoulder_yaw.pos": 0.0,
            "zbot_right_elbow.pos": -90.0,
            "zbot_right_wrist.pos": 0.0,
            "hand_pinky.pos": 0.0,
            "hand_ring.pos": 0.0,
            "hand_middle.pos": 0.0,
            "hand_index.pos": 0.0,
            "hand_thumb.pos": 0.0,
            "hand_extra.pos": 0.0,
        }
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.get_observation()

    def get_current_state(self):
        obs = self.get_observation()
        # Extract state values in the order expected by the kbot config
        state_values = []
        # Left arm (5 joints)
        for joint in ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist"]:
            state_values.append(obs[f"zbot_{joint}.pos"])
        # Right arm (5 joints) 
        for joint in ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist"]:
            state_values.append(obs[f"zbot_{joint}.pos"])
        # Hand (6 joints)
        for finger in ["pinky", "ring", "middle", "index", "thumb", "extra"]:
            state_values.append(obs[f"hand_{finger}.pos"])
        
        return np.array(state_values)

    def get_current_images(self):
        obs = self.get_observation()
        images = {}
        if "left" in obs:
            # Convert BGR to RGB
            images["left"] = cv2.cvtColor(obs["left"], cv2.COLOR_BGR2RGB)
        if "right" in obs:
            # Convert BGR to RGB  
            images["right"] = cv2.cvtColor(obs["right"], cv2.COLOR_BGR2RGB)
        return images

    def set_target_state(self, target_state):
        self.robot.send_action(target_state)

    def disconnect(self):
        self.robot.disconnect()
        print("================> KBot Robot disconnected")

    def __del__(self):
        if hasattr(self, 'robot'):
            self.disconnect()


#################################################################################


class Gr00tKBotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the bottle and place it in the container.",
    ):
        self.language_instruction = language_instruction
        # 480, 640 for camera resolution
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img_left, img_right, state):
        # Debug prints to see actual shapes and types
        print(f"DEBUG: state shape: {state.shape}")
        print(f"DEBUG: img_left type: {type(img_left)}, shape: {img_left.shape if hasattr(img_left, 'shape') else 'no shape'}")
        print(f"DEBUG: img_right type: {type(img_right)}, shape: {img_right.shape if hasattr(img_right, 'shape') else 'no shape'}")
        print(f"DEBUG: Using correct training dimensions - 4+4+5=13 joints")
        
        # Convert images to the right format for torch serialization
        img_left_np = np.array(img_left, dtype=np.uint8)
        img_right_np = np.array(img_right, dtype=np.uint8)
        
        # Ensure contiguous arrays for torch
        img_left_np = np.ascontiguousarray(img_left_np)
        img_right_np = np.ascontiguousarray(img_right_np)
        
        print(f"DEBUG: Final img_left_np shape: {img_left_np.shape}, dtype: {img_left_np.dtype}")
        print(f"DEBUG: Final img_right_np shape: {img_right_np.shape}, dtype: {img_right_np.dtype}")
        
        obs_dict = {
            "video.left": img_left_np[np.newaxis, :, :, :],                    # Add batch dimension [1, H, W, C]
            "video.right": img_right_np[np.newaxis, :, :, :],                  # Add batch dimension [1, H, W, C]
            "state.left_arm": state[:5].astype(np.float64)[np.newaxis, :],     # Add time dimension [1, 5] - with wrist
            "state.right_arm": state[5:10].astype(np.float64)[np.newaxis, :],   # Add time dimension [1, 5] - with wrist
            "state.right_hand": state[10:16].astype(np.float64)[np.newaxis, :], # Add time dimension [1, 6] - with extra
            "annotation.human.task_description": [self.language_instruction],  # Make it a list
        }
        res = self.policy.get_action(obs_dict)
        return res

    def sample_action(self):
        obs_dict = {
            "video.left": np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.right": np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.left_arm": np.zeros(5),      # 5 joints - with wrist
            "state.right_arm": np.zeros(5),     # 5 joints - with wrist
            "state.right_hand": np.zeros(6),    # 6 joints - with extra
            "annotation.human.task_description": self.language_instruction,
        }
        return self.policy.get_action(obs_dict)

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is aligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def view_dual_img(img_left, img_right):
    """
    Display both camera views side by side
    """
    plt.close('all')  # Close all figures before creating new ones
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img_left)
    ax1.set_title("Left Camera")
    ax1.axis("off")
    ax2.imshow(img_right)
    ax2.set_title("Right Camera")
    ax2.axis("off")
    plt.pause(0.001)
    plt.close(fig)  # Close this specific figure


#################################################################################

if __name__ == "__main__":
    import argparse

    default_dataset_path = "./demo_data/pick_bottle/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument("--cam_left_idx", type=int, default=0)
    parser.add_argument("--cam_right_idx", type=int, default=1) 
    parser.add_argument("--serial_port", type=str, default="/dev/ttyUSB0")
    parser.add_argument(
        "--lang_instruction", type=str, default="Pick up the bottle and place it in the container."
    )
    parser.add_argument("--record_imgs", action="store_true")
    args = parser.parse_args()

    # print lang_instruction
    print("lang_instruction: ", args.lang_instruction)

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = args.action_horizon
    MODALITY_KEYS = ["left_arm", "right_arm", "right_hand"]
    
    if USE_POLICY:
        client = Gr00tKBotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction=args.lang_instruction,
        )

        if args.record_imgs:
            # create a folder to save the images and delete all the images in the folder
            os.makedirs("eval_images", exist_ok=True)
            for file in os.listdir("eval_images"):
                os.remove(os.path.join("eval_images", file))

        robot = KBotRobot(
            calibrate=False,
            enable_cameras=True,
            cam_left_idx=args.cam_left_idx,
            cam_right_idx=args.cam_right_idx,
            serial_port=args.serial_port
        )
        image_count = 0
        
        with robot.activate():
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                images = robot.get_current_images()
                if "left" in images and "right" in images:
                    view_dual_img(images["left"], images["right"])
                    img_left = images["left"]
                    img_right = images["right"]
                else:
                    print("Warning: Could not get both camera images")
                    continue
                
                state = robot.get_current_state()
                action = client.get_action(img_left, img_right, state)
                start_time = time.time()
                
                for j in range(ACTION_HORIZON):
                    # Construct action dict for the robot
                    target_state = {}
                    
                    # Left arm actions (5 joints - with wrist)
                    left_arm_action = action["action.left_arm"][j]
                    for k, joint in enumerate(["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist"]):
                        target_state[f"zbot_{joint}.pos"] = left_arm_action[k]
                    
                    # Right arm actions (5 joints - with wrist)
                    right_arm_action = action["action.right_arm"][j]
                    for k, joint in enumerate(["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist"]):
                        target_state[f"zbot_{joint}.pos"] = right_arm_action[k]
                    
                    # Hand actions (6 joints - with extra)
                    hand_action = action["action.right_hand"][j]
                    for k, finger in enumerate(["pinky", "ring", "middle", "index", "thumb", "extra"]):
                        target_state[f"hand_{finger}.pos"] = hand_action[k]
                    
                    robot.set_target_state(target_state)
                    time.sleep(0.02)

                    # get the realtime image
                    images = robot.get_current_images()
                    if "left" in images and "right" in images:
                        view_dual_img(images["left"], images["right"])

                        if args.record_imgs:
                            # resize and save both images
                            img_left_small = cv2.resize(cv2.cvtColor(images["left"], cv2.COLOR_RGB2BGR), (320, 240))
                            img_right_small = cv2.resize(cv2.cvtColor(images["right"], cv2.COLOR_RGB2BGR), (320, 240))
                            cv2.imwrite(f"eval_images/img_left_{image_count}.jpg", img_left_small)
                            cv2.imwrite(f"eval_images/img_right_{image_count}.jpg", img_right_small)
                            image_count += 1

                    # 0.02*16 = 0.32 seconds per action chunk
                    print("executing action", j, "time taken", time.time() - start_time)
                print("Action chunk execution time taken", time.time() - start_time)
    else:
        # Test Dataset playback
        dataset = LeRobotDataset(
            repo_id="",
            root=args.dataset_path,
        )

        robot = KBotRobot(
            calibrate=False,
            enable_cameras=True,
            cam_left_idx=args.cam_left_idx,
            cam_right_idx=args.cam_right_idx,
            serial_port=args.serial_port
        )

        with robot.activate():
            print("Run replay of the dataset")
            actions = []
            for i in tqdm(range(min(ACTIONS_TO_EXECUTE, len(dataset))), desc="Loading actions"):
                try:
                    data_point = dataset[i]
                    action = data_point["action"]
                    
                    # Get dataset images
                    img_left = data_point["observation.images.left"].data.numpy()
                    img_right = data_point["observation.images.right"].data.numpy()
                    
                    # Convert from (C, H, W) to (H, W, C) and display
                    img_left = img_left.transpose(1, 2, 0)
                    img_right = img_right.transpose(1, 2, 0)
                    
                    # Get current robot images for comparison
                    robot_images = robot.get_current_images()
                    
                    if "left" in robot_images and "right" in robot_images:
                        # Show dataset image and overlay with robot image
                        view_dual_img(img_left, img_right)
                        # Could also overlay: view_img(img_left, robot_images["left"])
                    
                    # Convert action to robot format
                    target_state = {}
                    
                    # Map action array to robot joints (assuming action is [16] array)
                    # Left arm (5 joints)
                    for j, joint in enumerate(["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist"]):
                        target_state[f"zbot_{joint}.pos"] = action[j]
                    
                    # Right arm (5 joints)
                    for j, joint in enumerate(["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist"]):
                        target_state[f"zbot_{joint}.pos"] = action[j + 5]
                    
                    # Hand (6 joints)
                    for j, finger in enumerate(["pinky", "ring", "middle", "index", "thumb", "extra"]):
                        target_state[f"hand_{finger}.pos"] = action[j + 10]
                    
                    actions.append(action)
                    robot.set_target_state(target_state)
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"Error processing dataset item {i}: {e}")
                    continue

            # plot the actions
            if actions:
                plt.figure(figsize=(12, 8))
                actions_array = np.array(actions)
                for j in range(actions_array.shape[1]):
                    plt.subplot(4, 4, j + 1)
                    plt.plot(actions_array[:, j])
                    plt.title(f"Joint {j}")
                plt.tight_layout()
                plt.show()

            print("Done all actions")
            robot.go_home()
            print("Done home")
