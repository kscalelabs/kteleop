#!/usr/bin/env python3
"""
Example script demonstrating how to use the ZBot robot class with kos-kbot.

This script shows how to:
1. Connect to the ZBot robot
2. Get joint positions
3. Send joint commands
4. Monitor joint states in real-time

Make sure your kos-kbot is running before executing this script.
"""

import asyncio
import time
import math
from pathlib import Path

from lerobot.robots import ZBot, ZBotConfig


def main():
    """Main example function."""
    
    # Create ZBot configuration
    config = ZBotConfig(
        id="zbot_001",
        host="localhost",
        port=50051,
        # You can customize joint IDs if needed
        left_arm_ids=[11, 12, 13, 14, 15],
        right_arm_ids=[21, 22, 23, 24, 25],
    )
    
    # Create ZBot instance
    robot = ZBot(config)
    
    try:
        print("Connecting to ZBot...")
        robot.connect(calibrate=True)
        print("✅ Connected to ZBot!")
        
        # Get initial observation
        print("\n📊 Initial joint positions:")
        observation = robot.get_observation()
        for key, value in observation.items():
            if key.endswith('.pos'):
                print(f"  {key}: {value:.4f} rad")
        
        # Example 1: Oscillate left shoulder pitch
        print("\n🤖 Oscillating left shoulder pitch...")
        
        # Oscillate for 10 seconds
        start_time = time.time()
        duration = 10.0  # seconds
        frequency = 0.1  # Hz (oscillations per second)
        amplitude = 4  # radians (~5.7 degrees)
        
        while time.time() - start_time < duration:
            # Calculate oscillating position
            elapsed = time.time() - start_time
            oscillation = amplitude * math.sin(2 * math.pi * frequency * elapsed)
            
            # Send command
            action = {
                "left_shoulder_pitch.pos": oscillation
            }
            result = robot.send_action(action)
            
            # Get current position
            observation = robot.get_observation()
            current_pos = observation.get("left_shoulder_pitch.pos", 0.0)
            
            # Print status
            print(f"\rOscillating... Time: {elapsed:.1f}s | Target: {oscillation:.3f} | Current: {current_pos:.3f}", end="")
            
            # Small delay for smooth motion
            time.sleep(0.1)
        
        print(f"\n✅ Oscillation completed!")
        
        # Example 2: Move to a stable position
        print("\n🤖 Moving to stable position...")
        action = {
            "left_shoulder_pitch.pos": 0.0,
            "right_shoulder_pitch.pos": 0.0,
            "left_elbow.pos": 0.0,
            "right_elbow.pos": 0.0,
        }
        result = robot.send_action(action)
        print(f"Commands sent: {result}")
        
        # Wait and check final position
        time.sleep(2.0)
        print("\n📊 Final joint positions:")
        observation = robot.get_observation()
        for key, value in observation.items():
            if key.endswith('.pos'):
                print(f"  {key}: {value:.4f} rad")
        
        # Example 3: Real-time monitoring
        print("\n📈 Real-time monitoring (5 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            observation = robot.get_observation()
            print(f"\rTime: {time.time() - start_time:.1f}s | ", end="")
            for key, value in observation.items():
                if key.endswith('.pos'):
                    print(f"{key}: {value:.3f} ", end="")
            time.sleep(0.1)
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("Disconnecting...")
        robot.disconnect()
        print("✅ Disconnected")


if __name__ == "__main__":
    main() 