
"""Adapt existing PyBullet robot sensors to VisionNav Observation."""

from pathlib import Path
import math

import cv2
import numpy as np
import pybullet as p

from visionnav.core.types import (
    RobotState,
    Observation,
    CameraCalibration,
)


class PyBulletObservationAdapter:
    def __init__(
        self,
        robot,
        output_dir="logs/p1_observations",
        width=160,
        height=120,
    ):
        self.robot = robot
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.width = width
        self.height = height

    def _capture_camera(self):
        """Capture RGB, metric depth, raw mask and calibration."""

        # Reuse original robot's camera settings.
        near = float(self.robot.camera_near)
        far = float(self.robot.camera_far)
        fov_deg = float(self.robot.camera_fov)

        pos, orn = p.getBasePositionAndOrientation(
            self.robot.robot_id
        )

        rotation = np.asarray(
            p.getMatrixFromQuaternion(orn),
            dtype=np.float64,
        ).reshape(3, 3)

        pos = np.asarray(pos, dtype=np.float64)

        # Same default camera offsets as DiffDriveRobot.
        camera_offset = np.array([0.1, 0.0, 0.11])
        target_offset = np.array([1.0, 0.0, 0.05])

        eye = pos + rotation @ camera_offset
        target = pos + rotation @ target_offset

        forward = target - eye
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0])

        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)

        down = np.cross(forward, right)

        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=world_up.tolist(),
        )

        projection = p.computeProjectionMatrixFOV(
            fov=fov_deg,
            aspect=self.width / self.height,
            nearVal=near,
            farVal=far,
        )

        _, _, rgba, depth_buffer, mask = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb = np.asarray(rgba, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )[:, :, :3].copy()

        depth_buffer = np.asarray(
            depth_buffer, dtype=np.float32
        ).reshape(self.height, self.width)

        # Convert OpenGL depth buffer to metric depth.
        depth_m = (
            far * near
            / (far - (far - near) * depth_buffer)
        ).astype(np.float32)

        mask = np.asarray(mask, dtype=np.int32).reshape(
            self.height, self.width
        )

        # Optical camera frame: X right, Y down, Z forward.
        T = np.eye(4, dtype=np.float64)
        T[:3, 0] = right
        T[:3, 1] = down
        T[:3, 2] = forward
        T[:3, 3] = eye

        fy = self.height / (
            2 * math.tan(math.radians(fov_deg) / 2)
        )
        fx = fy

        calibration = CameraCalibration(
            width=self.width,
            height=self.height,
            fx=float(fx),
            fy=float(fy),
            cx=(self.width - 1) / 2,
            cy=(self.height - 1) / 2,
            T_world_camera=tuple(float(v) for v in T.flatten()),
        )

        return rgb, depth_m, mask, calibration

    def capture(
        self,
        timestamp_s: float,
        scene_id: str,
        task_id: str,
        frame_index: int,
    ) -> Observation:
        """Capture one synchronized simulation observation."""

        pose = self.robot.get_pose()
        velocity = self.robot.get_velocity()
        lidar = self.robot.get_lidar_scan()

        # Actual planar forward velocity, not commanded speed.
        yaw = float(pose[2])
        forward_velocity = (
            float(velocity[0]) * math.cos(yaw)
            + float(velocity[1]) * math.sin(yaw)
        )

        state = RobotState(
            time_s=timestamp_s,
            x_m=float(pose[0]),
            y_m=float(pose[1]),
            yaw_rad=yaw,
            v_mps=forward_velocity,
            w_radps=float(velocity[2]),
            frame_id="world",
        )

        rgb, depth, mask, camera = self._capture_camera()

        if not np.all(np.isfinite(depth)):
            raise ValueError("Depth contains NaN or Inf")

        rgb_path = self.output_dir / f"rgb_{frame_index:06d}.png"
        depth_path = self.output_dir / f"depth_{frame_index:06d}.npy"
        mask_path = self.output_dir / f"mask_{frame_index:06d}.npy"

        saved = cv2.imwrite(
            str(rgb_path),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        if not saved:
            raise IOError(f"Could not save image: {rgb_path}")

        np.save(depth_path, depth)
        np.save(mask_path, mask)

        return Observation(
            timestamp_s=timestamp_s,
            scene_id=scene_id,
            task_id=task_id,
            state=state,
            lidar_m=tuple(float(x) for x in lidar),
            rgb_path=rgb_path.as_posix(),
            depth_path=depth_path.as_posix(),
            segmentation_path=mask_path.as_posix(),
            camera=camera,
        )
