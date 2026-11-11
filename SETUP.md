> 这是原作者基础仿真安装/平台排错说明；VisionNav-Edge最终ACT系统请先阅读 [README.md](README.md)。历史命令或本机路径可能不适用于最终入口。

# PyBullet Robot Simulation Setup

This project contains a PyBullet-based differential drive robot simulation with basic navigation capabilities, ready for RL training.

## Quick Setup

### 1. Create Virtual Environment (Recommended)

```bash
cd /Users/abhi/Desktop/Github_Projects/pybullet_sim
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import pybullet; print('PyBullet installed successfully')"
python -c "import pybullet_data; print('Data path:', pybullet_data.getDataPath())"
```

## Running the Simulation

### Test Basic Robot Movement

```bash
python scripts/01_test_pybullet.py
```

This will:
- Load the differential drive robot
- Test basic forward/backward and turning motions
- Display the robot in PyBullet GUI

### Test Simple Navigation

```bash
python scripts/02_simple_navigation.py
```

This will:
- Create a robot at origin (0, 0)
- Set a goal at (4, 4)
- Use a simple proportional controller to navigate to the goal
- Show the robot moving toward the green sphere (goal marker)

### Test Sensor Streams (NEW!)

```bash
python scripts/03_sensor_streams.py
```

This will:
- Display real-time sensor visualizations:
  - **Lidar Scan**: 2D top-down view with colored rays
  - **RGB Camera**: Live camera feed from robot's perspective
  - **Depth Camera**: Depth map visualization
  - **IMU Data**: Accelerometer and gyroscope readings
  - **Odometry**: Wheel velocities and robot velocity
  - **Trajectory**: Path the robot has traveled

### Record Sensor Data

```bash
python scripts/04_sensor_logger.py
```

This will:
- Record all sensor data to JSON files
- Save camera images (RGB and depth)
- Create timestamped logs in `logs/sensor_data/`
- Useful for data analysis and RL training preparation

## Project Structure

```
pybullet_sim/
├── urdf/                    # Robot URDF files
│   └── diff_drive_robot.urdf
├── scripts/                 # Python scripts
│   ├── diff_drive_robot.py  # Robot controller class
│   ├── 01_test_pybullet.py  # Basic movement test
│   └── 02_simple_navigation.py  # Navigation test
├── models/                  # Trained RL models (created during training)
├── logs/                    # Training logs and plots
├── configs/                 # Configuration files
├── assets/                  # Additional assets
└── requirements.txt         # Python dependencies
```

## Robot Specifications

- **Type**: Differential drive robot
- **Wheel radius**: 0.05 m
- **Wheel base**: 0.22 m
- **Max linear velocity**: 0.5 m/s
- **Max angular velocity**: 2.0 rad/s

### Sensors

- **Lidar**: 36-ray 2D lidar, 5m range
- **RGB Camera**: 320x240 resolution, 60° FOV
- **Depth Camera**: Same resolution, 0.01-10m range
- **IMU**: Simulated accelerometer and gyroscope (with noise)
- **Odometry**: Wheel encoder simulation

## Next Steps

Once basic simulation is working:
1. Test navigation with obstacles
2. Set up Gymnasium environment for RL
3. Train RL agent using stable-baselines3

## Troubleshooting

**PyBullet GUI doesn't open:**
- Try updating PyBullet: `pip install --upgrade pybullet`
- On macOS, ensure you have XQuartz installed if needed
- On Linux headless servers, use `p.DIRECT` mode instead of `p.GUI`

**Import errors:**
- Make sure you're in the project root directory
- Ensure virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

