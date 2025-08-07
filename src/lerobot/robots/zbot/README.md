# ZBot Robot Interface

The ZBot robot class provides a LeRobot-compatible interface for controlling the upper body joints of the kos-kbot platform.

## Overview

ZBot interfaces with the kos-kbot's KOS platform to control the robot's arms. It provides:

- **Joint Control**: Control over left and right arm joints (IDs 11-15 and 21-25)
- **Position Monitoring**: Real-time joint position and velocity feedback
- **Safety Limits**: Built-in safety limits to prevent dangerous movements
- **Calibration**: Automatic joint calibration and offset management
- **Camera Support**: Optional camera integration for visual feedback

## Joint Configuration

### Left Arm (IDs 11-15)
- `left_shoulder_pitch` (ID 11): Shoulder pitch joint
- `left_shoulder_roll` (ID 12): Shoulder roll joint  
- `left_shoulder_yaw` (ID 13): Shoulder yaw joint
- `left_elbow` (ID 14): Elbow joint
- `left_wrist` (ID 15): Wrist joint

### Right Arm (IDs 21-25)
- `right_shoulder_pitch` (ID 21): Shoulder pitch joint
- `right_shoulder_roll` (ID 22): Shoulder roll joint
- `right_shoulder_yaw` (ID 23): Shoulder yaw joint
- `right_elbow` (ID 24): Elbow joint
- `right_wrist` (ID 25): Wrist joint

## Installation

1. **Install LeRobot** (if not already installed):
   ```bash
   cd /path/to/lerobot
   pip install -e .
   ```

2. **Install pykos** (KOS Python client):
   ```bash
   pip install pykos
   ```

3. **Start kos-kbot**:
   ```bash
   cd /path/to/kos-kbot
   cargo run
   ```

## Usage

### Basic Usage

```python
from lerobot.robots import ZBot, ZBotConfig

# Create configuration
config = ZBotConfig(
    id="zbot_001",
    host="localhost",
    port=50051,
)

# Create and connect robot
robot = ZBot(config)
robot.connect(calibrate=True)

# Get current joint positions
observation = robot.get_observation()
print("Joint positions:", observation)

# Send joint commands
action = {
    "left_shoulder_pitch.pos": 0.2,  # radians
    "right_elbow.pos": -0.3,
}
result = robot.send_action(action)

# Disconnect
robot.disconnect()
```

### Configuration Options

```python
config = ZBotConfig(
    id="zbot_001",
    host="localhost",
    port=50051,
    
    # Customize joint IDs if needed
    left_arm_ids=[11, 12, 13, 14, 15],
    right_arm_ids=[21, 22, 23, 24, 25],
    
    # Safety limits
    max_velocity=7200.0,  # rad/s
    max_position_change=0.5,  # rad
    
    # Control rates
    command_rate_hz=100.0,
    update_rate_hz=100.0,
)
```

### Observation Structure

The robot returns observations with the following structure:

```python
{
    # Joint positions (radians)
    "left_shoulder_pitch.pos": float,
    "left_shoulder_roll.pos": float,
    "left_shoulder_yaw.pos": float,
    "left_elbow.pos": float,
    "left_wrist.pos": float,
    "right_shoulder_pitch.pos": float,
    "right_shoulder_roll.pos": float,
    "right_shoulder_yaw.pos": float,
    "right_elbow.pos": float,
    "right_wrist.pos": float,
    
    # Joint velocities (rad/s)
    "left_shoulder_pitch.vel": float,
    "left_shoulder_roll.vel": float,
    # ... (same pattern for all joints)
    
    # Camera images (if configured)
    "camera_name": numpy.ndarray,  # (height, width, 3)
}
```

### Action Structure

Actions should specify target positions for joints:

```python
{
    "left_shoulder_pitch.pos": 0.2,   # Target position in radians
    "right_elbow.pos": -0.3,
    "left_wrist.pos": 0.1,
    # ... specify only the joints you want to move
}
```

## Safety Features

- **Position Limits**: Maximum position change per command (default: 0.5 rad)
- **Velocity Limits**: Maximum joint velocity (default: 7200 rad/s)
- **Connection Validation**: Ensures KOS connection before operations
- **Error Handling**: Graceful handling of communication errors

## Examples

See `examples/zbot_example.py` for a complete working example.

## Troubleshooting

### Connection Issues
- Ensure kos-kbot is running: `cargo run` in kos-kbot directory
- Check KOS port: Default is 50051
- Verify pykos installation: `pip install pykos`

### Joint Control Issues
- Check joint IDs match your kos-kbot configuration
- Verify joint limits in kos-kbot's `lib.rs`
- Monitor KOS logs for actuator errors

### Performance Issues
- Adjust `command_rate_hz` and `update_rate_hz` in config
- Consider reducing `max_position_change` for smoother motion
- Monitor system resources during operation

## Integration with LeRobot

ZBot follows the standard LeRobot interface, making it compatible with:

- **Teleoperation**: Use LeRobot's teleoperation tools
- **Policy Training**: Train RL policies with LeRobot
- **Data Collection**: Record demonstrations and episodes
- **Evaluation**: Use LeRobot's evaluation frameworks

## License

This implementation follows the same license as LeRobot (Apache 2.0). 