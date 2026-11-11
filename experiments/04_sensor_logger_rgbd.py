
#!/usr/bin/env python3
"""Record synchronized RGB-D and robot sensor data from PyBullet."""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

# Allow imports from the project root, regardless of how the script is started.
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diff_drive_robot import DiffDriveRobot


# ===========================
# 1. Configuration
# ===========================

PHYSICS_HZ = 240
SIM_STEPS_PER_CONTROL = 10
TOTAL_CONTROL_STEPS = 300
LOG_INTERVAL = 5

OUTPUT_ROOT = PROJECT_ROOT / "logs" / "sensor_data_rgbd"

ROBOT_URDF = (
    PROJECT_ROOT / "urdf" / "diff_drive_robot.urdf"
)


# ===========================
# 2. Sensor data logger
# ===========================

class SensorDataLogger:
    """Save RGB, depth and other robot observations."""

    def __init__(self, output_root=OUTPUT_ROOT):
        # Separate output directory for each run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.log_dir = Path(output_root) / run_id
        self.log_dir.mkdir(parents=True, exist_ok=False)

        self.log_file = self.log_dir / "sensor_log.json"

        self.samples = []
        self.sample_count = 0

        self.metadata = {
            "run_id": run_id,
            "description": "PyBullet RGB-D robot sensor recording",
            "physics_hz": PHYSICS_HZ,
            "sim_steps_per_control": SIM_STEPS_PER_CONTROL,
            "log_interval": LOG_INTERVAL,
            "rgb_format": "PNG",
            "depth_format": "NPY float32",
            "depth_unit": "meter",
            "position_unit": "meter",
            "angle_unit": "radian",
            "linear_velocity_unit": "meter/second",
            "angular_velocity_unit": "radian/second",
            "observation_timing": "before_action",
            "odometry_note": (
                "Wheel velocities are derived from commands "
                "in the original DiffDriveRobot implementation."
            ),
            "imu_note": (
                "Original robot IMU uses a simplified "
                "acceleration model."
            ),
        }

        print(f"Recording directory: {self.log_dir}")

    def log_sample(
        self,
        sim_time,
        pose,
        lidar_scan,
        imu_data,
        odom_data,
        rgb_image,
        depth_image,
        action,
    ):
        """Save one synchronized pre-action observation."""

        index = self.sample_count

        rgb_name = f"rgb_{index:06d}.png"
        depth_name = f"depth_{index:06d}.npy"

        rgb_path = self.log_dir / rgb_name
        depth_path = self.log_dir / depth_name

        # Convert images into NumPy arrays
        rgb = np.asarray(rgb_image)
        depth = np.asarray(depth_image, dtype=np.float32)

        # Validate image shape
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"Invalid RGB image shape: {rgb.shape}"
            )

        if depth.ndim != 2:
            raise ValueError(
                f"Invalid depth image shape: {depth.shape}"
            )

        if rgb.shape[:2] != depth.shape:
            raise ValueError(
                "RGB and depth resolutions do not match"
            )

        if not np.isfinite(depth).all():
            raise ValueError(
                "Depth image contains NaN or infinity"
            )

        if (depth <= 0).any():
            raise ValueError(
                "Depth image contains non-positive values"
            )

        # PyBullet RGB -> OpenCV BGR
        rgb_bgr = cv2.cvtColor(
            rgb.astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )

        # Save RGB image
        saved = cv2.imwrite(str(rgb_path), rgb_bgr)

        if not saved:
            raise IOError(
                f"Failed to save RGB image: {rgb_path}"
            )

        # Save depth without losing floating-point precision
        np.save(depth_path, depth)

        # Convert LiDAR results
        lidar_distances = np.asarray(
            lidar_scan, dtype=float
        )

        # Build sample metadata
        sample = {
            "frame_id": index,
            "timestamp": float(sim_time),

            "pose": {
                "x": float(pose[0]),
                "y": float(pose[1]),
                "theta": float(pose[2]),
            },

            "lidar": {
                "distances": lidar_distances.tolist(),
                "num_rays": int(len(lidar_distances)),
                "max_range": 5.0,
            },

            "imu": {
                "accel": {
                    "x": float(imu_data["accel"][0]),
                    "y": float(imu_data["accel"][1]),
                    "z": float(imu_data["accel"][2]),
                },

                "gyro": {
                    "x": float(imu_data["gyro"][0]),
                    "y": float(imu_data["gyro"][1]),
                    "z": float(imu_data["gyro"][2]),
                },

                "orientation": {
                    "roll": float(
                        imu_data["orientation"][0]
                    ),
                    "pitch": float(
                        imu_data["orientation"][1]
                    ),
                    "yaw": float(
                        imu_data["orientation"][2]
                    ),
                },
            },

            "odometry": {
                "left_wheel_vel": float(
                    odom_data["left_wheel_vel"]
                ),
                "right_wheel_vel": float(
                    odom_data["right_wheel_vel"]
                ),
                "velocity": np.asarray(
                    odom_data["velocity"],
                    dtype=float
                ).tolist(),
            },

            "action": {
                "linear_vel": float(action[0]),
                "angular_vel": float(action[1]),
            },

            "rgb_image": rgb_name,
            "depth_image": depth_name,
        }

        self.samples.append(sample)
        self.sample_count += 1

    def save(self):
        """Write the JSON index and summary information."""

        data = {
            "metadata": self.metadata,
            "samples": self.samples,
        }

        with self.log_file.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"\nJSON saved: {self.log_file}")
        print(f"Total samples: {self.sample_count}")

        return self.log_file

    def get_summary(self):
        """Calculate summary statistics."""

        if not self.samples:
            return {
                "total_samples": 0,
                "duration": 0.0,
                "distance_traveled": 0.0,
            }

        poses = np.array([
            [
                s["pose"]["x"],
                s["pose"]["y"]
            ]
            for s in self.samples
        ])

        timestamps = np.array([
            s["timestamp"]
            for s in self.samples
        ])

        distances = np.linalg.norm(
            np.diff(poses, axis=0),
            axis=1
        )

        return {
            "total_samples": len(self.samples),
            "duration": float(
                timestamps[-1] - timestamps[0]
            ),
            "distance_traveled": float(
                distances.sum()
            ),
            "start_position": poses[0].tolist(),
            "end_position": poses[-1].tolist(),
        }


# ===========================
# 3. Robot motion commands
# ===========================

def get_action(step):
    """Generate a simple sequence of movement commands."""

    if step < 100:
        return 0.3, 0.0

    elif step < 200:
        return 0.2, -1.0

    else:
        return 0.2, 1.0


# ===========================
# 4. Main simulation
# ===========================

def main():
    print("=" * 50)
    print("VisionNav-Edge RGB-D Sensor Recorder")
    print("=" * 50)

    # Check robot model
    if not ROBOT_URDF.exists():
        raise FileNotFoundError(
            f"Robot URDF not found: {ROBOT_URDF}"
        )

    # Start PyBullet
    client = p.connect(p.GUI)

    if client < 0:
        raise RuntimeError(
            "Failed to connect to PyBullet GUI"
        )

    logger = None
    physics_steps = 0

    try:
        # Initialize simulation
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / PHYSICS_HZ)

        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )

        # Load ground
        p.loadURDF("plane.urdf")

        # Load robot
        robot = DiffDriveRobot(
            str(ROBOT_URDF),
            start_pos=[0, 0, 0.1]
        )

        # Create logger
        logger = SensorDataLogger()

        print("\nRobot loaded successfully")
        print("Recording sensor data...")
        print("Press Ctrl+C to stop and save\n")

        # Main control loop
        for step in range(TOTAL_CONTROL_STEPS):

            # Determine the action to execute next
            v, w = get_action(step)

            # Record observation before executing this action
            if step % LOG_INTERVAL == 0:

                sim_time = physics_steps / PHYSICS_HZ

                # Read robot state and sensors
                pose = robot.get_pose()
                lidar_scan = robot.get_lidar_scan()
                imu_data = robot.get_imu_data()
                odom_data = robot.get_odometry()

                # Capture RGB and depth at the same pose
                rgb_image, depth_image = (
                    robot.get_camera_image()
                )

                # Save one sample
                logger.log_sample(
                    sim_time=sim_time,
                    pose=pose,
                    lidar_scan=lidar_scan,
                    imu_data=imu_data,
                    odom_data=odom_data,
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    action=[v, w],
                )

            # Execute the action after recording observation
            robot.set_velocity(v, w)

            # Advance physics simulation
            for _ in range(SIM_STEPS_PER_CONTROL):
                p.stepSimulation()
                physics_steps += 1

            if step % 50 == 0:
                print(
                    f"Step {step:03d} | "
                    f"Samples: {logger.sample_count} | "
                    f"Sim time: "
                    f"{physics_steps / PHYSICS_HZ:.2f}s"
                )

            # Slow down only for visual observation
            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        print("\nRecording interrupted by user")

    finally:
        # Always save collected samples and disconnect
        try:
            if logger is not None:

                logger.save()

                summary = logger.get_summary()

                print("\n" + "=" * 50)
                print("Recording Summary")
                print("=" * 50)

                print(
                    "Total samples:",
                    summary["total_samples"]
                )

                print(
                    "Duration:",
                    round(summary["duration"], 2),
                    "simulation seconds"
                )

                print(
                    "Distance traveled:",
                    round(summary["distance_traveled"], 2),
                    "m"
                )

                print(
                    "Start position:",
                    summary.get("start_position")
                )

                print(
                    "End position:",
                    summary.get("end_position")
                )

                rgb_files = list(
                    logger.log_dir.glob("rgb_*.png")
                )

                depth_files = list(
                    logger.log_dir.glob("depth_*.npy")
                )

                print("RGB files:", len(rgb_files))
                print("Depth files:", len(depth_files))

                print(
                    "\nData saved to:",
                    logger.log_dir
                )

                expected = TOTAL_CONTROL_STEPS // LOG_INTERVAL

                if (
                    logger.sample_count == expected
                    and len(rgb_files) == expected
                    and len(depth_files) == expected
                ):
                    print("\nRGB-D RECORDING PASSED")
                else:
                    print(
                        "\nRecording incomplete or interrupted; "
                        "verify the saved samples."
                    )

        finally:
            if p.isConnected():
                p.disconnect()


if __name__ == "__main__":
    main()
