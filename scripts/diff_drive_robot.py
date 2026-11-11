#!/usr/bin/env python3
"""Differential drive robot controller for PyBullet."""
import pybullet as p
import numpy as np

class DiffDriveRobot:
    """Differential drive robot with sensors and control."""
    
    def __init__(self, urdf_path, start_pos=[0, 0, 0.1], start_yaw=0):
        """
        Initialize robot in PyBullet simulation.
        
        Args:
            urdf_path: Path to robot URDF file
            start_pos: Initial [x, y, z] position
            start_yaw: Initial heading (radians)
        """
        self.urdf_path = urdf_path
        
        # Robot parameters
        self.wheel_radius = 0.05
        self.wheel_base = 0.22  # distance between wheels
        self.max_wheel_vel = 10.0  # rad/s
        self.max_force = 10.0
        
        # Load robot
        start_orn = p.getQuaternionFromEuler([0, 0, start_yaw])
        self.robot_id = p.loadURDF(urdf_path, start_pos, start_orn)
        
        # Find wheel joints
        self._find_joints()
        
        # Lidar parameters
        self.lidar_rays = 36
        self.lidar_range = 5.0
        self.lidar_height = 0.11
        
        # Camera parameters
        self.camera_width = 320
        self.camera_height = 240
        self.camera_fov = 60
        self.camera_near = 0.01
        self.camera_far = 10.0
        
        # IMU parameters (simulated)
        self.imu_noise_std = 0.01
        
        # State
        self.last_v = 0.0
        self.last_w = 0.0
    
    def _find_joints(self):
        """Find wheel joint indices."""
        self.left_wheel_joint = None
        self.right_wheel_joint = None
        
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode('utf-8')
            
            if 'left_wheel' in name:
                self.left_wheel_joint = i
            elif 'right_wheel' in name:
                self.right_wheel_joint = i
        
        if self.left_wheel_joint is None or self.right_wheel_joint is None:
            raise ValueError("Could not find wheel joints in URDF")
    
    def get_pose(self):
        """Get current robot pose [x, y, theta]."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        return np.array([pos[0], pos[1], euler[2]])
    
    def get_velocity(self):
        """Get current robot velocity [vx, vy, w]."""
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        return np.array([lin_vel[0], lin_vel[1], ang_vel[2]])
    
    def set_velocity(self, v, w):
        """
        Set robot velocity using differential drive kinematics.
        
        Args:
            v: Linear velocity (m/s)
            w: Angular velocity (rad/s)
        """
        self.last_v = v
        self.last_w = w
        
        # Convert to wheel velocities
        # v = (v_r + v_l) * r / 2
        # w = (v_r - v_l) * r / L
        v_r = (2*v + w*self.wheel_base) / (2*self.wheel_radius)
        v_l = (2*v - w*self.wheel_base) / (2*self.wheel_radius)
        
        # Clamp wheel velocities
        v_r = np.clip(v_r, -self.max_wheel_vel, self.max_wheel_vel)
        v_l = np.clip(v_l, -self.max_wheel_vel, self.max_wheel_vel)
        
        # Apply to motors
        p.setJointMotorControl2(
            self.robot_id, self.left_wheel_joint,
            p.VELOCITY_CONTROL,
            targetVelocity=v_l,
            force=self.max_force
        )
        p.setJointMotorControl2(
            self.robot_id, self.right_wheel_joint,
            p.VELOCITY_CONTROL,
            targetVelocity=v_r,
            force=self.max_force
        )
    
    def get_lidar_scan(self):
        """
        Simulate 2D lidar scan using raycasting.
        
        Returns:
            Array of distances for each ray
        """
        pose = self.get_pose()
        ray_from = []
        ray_to = []
        
        for i in range(self.lidar_rays):
            angle = pose[2] + (2*np.pi * i / self.lidar_rays)
            
            start = [
                pose[0],
                pose[1],
                self.lidar_height
            ]
            end = [
                pose[0] + self.lidar_range * np.cos(angle),
                pose[1] + self.lidar_range * np.sin(angle),
                self.lidar_height
            ]
            
            ray_from.append(start)
            ray_to.append(end)
        
        # Batch raycast
        results = p.rayTestBatch(ray_from, ray_to)
        
        # Extract distances
        distances = []
        for result in results:
            hit_fraction = result[2]
            distance = hit_fraction * self.lidar_range
            distances.append(distance)
        
        return np.array(distances)
    
    def check_collision(self):
        """Check if robot is in collision with any object."""
        try:
            contacts = p.getContactPoints(bodyA=self.robot_id)
            
            if not contacts:
                return False
            
            for contact in contacts:
                # Ignore ground contact
                if contact[2] == 0:  # ground plane ID
                    continue
                # Check if contact is with base (not wheels)
                # contact[3] is link index, -1 is base link
                if contact[3] == -1:  # base link
                    # Check contact force - only consider significant collisions
                    if len(contact) > 9 and contact[9] > 0.1:  # Normal force
                        return True
        except:
            # If we can't check contacts, assume no collision
            return False
        
        return False
    
    def get_camera_image(self, camera_pos_offset=[0.1, 0, 0.11], look_at_offset=[1.0, 0, 0.05]):
        """
        Get RGB and depth images from robot's camera.
        
        Args:
            camera_pos_offset: Camera position relative to robot base [x, y, z]
            look_at_offset: Look-at point relative to robot base [x, y, z]
            
        Returns:
            rgb_image: RGB image array (height, width, 3)
            depth_image: Depth image array (height, width)
        """
        pose = self.get_pose()
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Calculate camera position in world frame
        # Rotate offset by robot orientation
        rot_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        camera_pos_world = np.array(pos) + rot_matrix @ np.array(camera_pos_offset)
        look_at_world = np.array(pos) + rot_matrix @ np.array(look_at_offset)
        
        # Camera up vector (always pointing up)
        up_vector = [0, 0, 1]
        
        # Get camera view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos_world,
            cameraTargetPosition=look_at_world,
            cameraUpVector=up_vector
        )
        
        # Projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        
        # Get camera image
        _, _, rgb, depth, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert RGB from (H, W, 4) to (H, W, 3) and normalize
        rgb_image = np.array(rgb)[:, :, :3]
        
        # Convert depth buffer to actual depth
        depth_image = np.array(depth)
        depth_image = self.camera_far * self.camera_near / (self.camera_far - (self.camera_far - self.camera_near) * depth_image)
        
        return rgb_image, depth_image
    
    def get_imu_data(self):
        """
        Get simulated IMU data (accelerometer, gyroscope).
        
        Returns:
            dict with 'accel' [ax, ay, az] and 'gyro' [gx, gy, gz]
        """
        pose = self.get_pose()
        vel = self.get_velocity()
        
        # Get orientation in quaternion
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Get linear and angular acceleration (simplified)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Simulate accelerometer (gravity + linear acceleration in body frame)
        gravity = np.array([0, 0, -9.81])
        rot_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # Transform gravity to body frame
        gravity_body = rot_matrix.T @ gravity
        
        # Add linear acceleration (simplified - just use velocity change)
        accel = gravity_body + np.array([0, 0, 0])  # Could add actual acceleration
        
        # Add noise
        accel += np.random.normal(0, self.imu_noise_std, 3)
        
        # Gyroscope (angular velocity in body frame)
        gyro = np.array([ang_vel[0], ang_vel[1], ang_vel[2]])
        gyro += np.random.normal(0, self.imu_noise_std, 3)
        
        return {
            'accel': accel,
            'gyro': gyro,
            'orientation': euler
        }
    
    def get_odometry(self):
        """
        Get odometry data (wheel encoders simulation).
        
        Returns:
            dict with 'left_wheel_vel', 'right_wheel_vel', 'pose', 'velocity'
        """
        # Get wheel velocities (simulated from commanded velocities)
        v = self.last_v
        w = self.last_w
        
        v_r = (2*v + w*self.wheel_base) / (2*self.wheel_radius)
        v_l = (2*v - w*self.wheel_base) / (2*self.wheel_radius)
        
        pose = self.get_pose()
        vel = self.get_velocity()
        
        return {
            'left_wheel_vel': v_l,
            'right_wheel_vel': v_r,
            'pose': pose,
            'velocity': vel
        }
    
    def reset(self, pos, yaw):
        """Reset robot to new position."""
        try:
            # Check if robot still exists
            try:
                p.getBasePositionAndOrientation(self.robot_id)
            except:
                # Robot was deleted, need to reload
                start_orn = p.getQuaternionFromEuler([0, 0, yaw])
                self.robot_id = p.loadURDF(self.urdf_path, [pos[0], pos[1], 0.1], start_orn)
                self._find_joints()
                return
            
            # Reset position and velocity
            orn = p.getQuaternionFromEuler([0, 0, yaw])
            p.resetBasePositionAndOrientation(self.robot_id, [pos[0], pos[1], 0.1], orn)
            p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
            
            # Reset wheel velocities
            p.setJointMotorControl2(
                self.robot_id, self.left_wheel_joint,
                p.VELOCITY_CONTROL, targetVelocity=0, force=0
            )
            p.setJointMotorControl2(
                self.robot_id, self.right_wheel_joint,
                p.VELOCITY_CONTROL, targetVelocity=0, force=0
            )
            
            # Reset state
            self.last_v = 0.0
            self.last_w = 0.0
        except Exception as e:
            print(f"Warning: Reset failed: {e}")

