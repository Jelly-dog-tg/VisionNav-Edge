#!/usr/bin/env python3
"""Gymnasium environment for PyBullet differential drive navigation."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import os
import importlib.util
import json
from typing import Dict, Tuple, Optional

# Dynamic import for modules with numeric names
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load required modules
env_builder_module = load_module("env_builder", 
    os.path.join(os.path.dirname(__file__), "05_environment_builder.py"))
EnvironmentBuilder = env_builder_module.EnvironmentBuilder

from scripts.diff_drive_robot import DiffDriveRobot


class PyBulletNavEnv(gym.Env):
    """Gymnasium environment for PyBullet differential drive navigation."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, env_type="warehouse", num_obstacles=None):
        """
        Initialize navigation environment.
        
        Args:
            render_mode: "human" (GUI) or None (headless)
            env_type: "warehouse", "street", or "maze"
            num_obstacles: Number of obstacles (None = default for env_type)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.env_type = env_type
        self.num_obstacles = num_obstacles
        
        # PyBullet setup
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # Action space: [v, w] - linear and angular velocity
        # Forward-only movement (no backward driving) - based on IR-SIM approach
        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0], dtype=np.float32),  # No negative linear velocity
            high=np.array([0.7, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space:
        # [dx, dy, dtheta, distance, v, w, lidar_readings (36 values), min_lidar]
        # NOTE: Added distance to help agent know when close to goal
        obs_dim = 6 + 36 + 1  # relative goal + distance + velocities + lidar + min_lidar
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Environment parameters
        self.arena_size = 10.0
        self.goal_threshold = 0.5  # 0.5m threshold (reasonable for robot radius ~0.1m)
        self.max_steps = 1500  # Increased from 1000 to give more time
        self.time_step = 1/24  # Control rate
        
        # State
        self.robot = None
        self.goal = None
        self.obstacles = []
        self.builder = None
        self.step_count = 0
        self.prev_distance = None
        self.start_pos = None
        self.recent_progress = []  # Track recent progress to detect stuck behavior
        self.stuck_threshold = 0.1  # Consider stuck if progress < 10cm over recent steps (was too sensitive at 2cm)
        
    def _create_environment(self):
        """Create or reset the simulation environment."""
        # Clear existing objects
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load ground
        p.loadURDF("plane.urdf")
        
        # Create environment
        if self.builder is None:
            self.builder = EnvironmentBuilder(self.physics_client)
        
        # Create obstacles based on environment type
        if self.env_type == "warehouse":
            self.obstacles = self.builder.create_warehouse_environment()
        elif self.env_type == "street":
            self.obstacles = self.builder.create_street_environment()
        elif self.env_type == "maze":
            self.obstacles = self.builder.create_simple_maze()
        else:
            # Default: simple obstacles
            self.obstacles = self.builder.create_warehouse_environment()
        
        # Random goal position (far from start, not in obstacles, not too close to walls)
        # Walls are at ±arena_size/2 (e.g., ±5.0 for arena_size=10.0)
        # Need enough clearance for robot to actually reach goal (robot radius ~0.3m + goal threshold 0.5m + safety)
        min_wall_clearance = 1.2  # Increased: robot needs space to reach goal (0.3m radius + 0.5m threshold + 0.4m safety)
        min_obstacle_clearance = 0.8  # Increased: ensure robot can actually reach goal (0.3m radius + 0.5m threshold)
        max_attempts = 100
        goal_found = False
        
        for attempt in range(max_attempts):
            # Place goal away from walls (walls are at ±arena_size/2)
            gx = np.random.uniform(-self.arena_size/2 + min_wall_clearance, self.arena_size/2 - min_wall_clearance)
            gy = np.random.uniform(-self.arena_size/2 + min_wall_clearance, self.arena_size/2 - min_wall_clearance)
            
            # Must be far from start
            if np.sqrt(gx**2 + gy**2) < 2.0:
                continue
            
            # Check if goal is too close to walls (walls are at ±arena_size/2)
            wall_distance = min(
                abs(gx - self.arena_size/2),   # Distance to right wall
                abs(gx + self.arena_size/2),   # Distance to left wall
                abs(gy - self.arena_size/2),   # Distance to top wall
                abs(gy + self.arena_size/2)     # Distance to bottom wall
            )
            if wall_distance < min_wall_clearance:
                continue
            
            # Check if goal is inside or too close to obstacles using raycasting
            # Cast rays from goal in multiple directions to check for nearby obstacles
            goal_valid = True
            num_rays = 8
            for angle in np.linspace(0, 2*np.pi, num_rays, endpoint=False):
                ray_end_x = gx + min_obstacle_clearance * np.cos(angle)
                ray_end_y = gy + min_obstacle_clearance * np.sin(angle)
                ray_start = [gx, gy, 0.2]  # Slightly above ground
                ray_end = [ray_end_x, ray_end_y, 0.2]
                
                # Raycast to check for obstacles (ignore ground plane with ID 0)
                ray_result = p.rayTest(ray_start, ray_end)
                if ray_result and ray_result[0][0] != -1 and ray_result[0][0] != 0:  # Hit something (not ground)
                    hit_fraction = ray_result[0][2]
                    if hit_fraction < 0.8:  # Obstacle too close
                        goal_valid = False
                        break
            
            # Also check directly above goal position for obstacles
            if goal_valid:
                # Cast ray straight up from goal to check if goal is inside an obstacle
                ray_start = [gx, gy, 0.1]
                ray_end = [gx, gy, 1.0]  # Check up to 1m high
                ray_result = p.rayTest(ray_start, ray_end)
                if ray_result and ray_result[0][0] != -1 and ray_result[0][0] != 0:
                    # Hit an obstacle directly above goal - goal might be inside obstacle
                    goal_valid = False
            
            if goal_valid:
                self.goal = np.array([gx, gy])
                goal_found = True
                
                # #region agent log - goal placement
                try:
                    with open('/Users/abhi/Desktop/Github_Projects/pybullet_sim/.cursor/debug.log', 'a') as f:
                        log_entry = json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "F",
                            "location": "pybullet_nav_env.py:_create_environment",
                            "message": "Goal placed",
                            "data": {
                                "goal_pos": [float(gx), float(gy)],
                                "wall_distance": float(wall_distance),
                                "attempt": int(attempt + 1),
                                "min_wall_clearance": float(min_wall_clearance),
                                "min_obstacle_clearance": float(min_obstacle_clearance)
                            },
                            "timestamp": int(os.times().elapsed * 1000)
                        }) + "\n"
                        f.write(log_entry)
                except: pass
                # #endregion
                break
        
        if not goal_found:
            # Fallback: place goal at center of arena if all attempts failed
            print("Warning: Could not find valid goal position, using default")
            self.goal = np.array([self.arena_size/4, self.arena_size/4])
            
            # #region agent log - goal placement fallback
            try:
                with open('/Users/abhi/Desktop/Github_Projects/pybullet_sim/.cursor/debug.log', 'a') as f:
                    log_entry = json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "F",
                        "location": "pybullet_nav_env.py:_create_environment",
                        "message": "Goal placement fallback",
                        "data": {
                            "goal_pos": [float(self.goal[0]), float(self.goal[1])],
                            "max_attempts": int(max_attempts)
                        },
                        "timestamp": int(os.times().elapsed * 1000)
                    }) + "\n"
                    f.write(log_entry)
            except: pass
            # #endregion
        
        # Create robot at origin
        urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "urdf", "diff_drive_robot.urdf")
        start_yaw = np.random.uniform(-np.pi, np.pi)
        self.start_pos = [0, 0, 0.1]
        self.robot = DiffDriveRobot(urdf_path, start_pos=self.start_pos, start_yaw=start_yaw)
        
        # Goal marker (for visualization)
        if self.render_mode == "human":
            goal_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.3,
                rgbaColor=[0.2, 0.8, 0.2, 0.8]
            )
            self.goal_marker = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=goal_visual,
                basePosition=[self.goal[0], self.goal[1], 0.3]
            )
    
    def _get_obs(self) -> np.ndarray:
        """Compute observation."""
        pose = self.robot.get_pose()
        
        # Relative goal position
        dx = self.goal[0] - pose[0]
        dy = self.goal[1] - pose[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        
        # Heading to goal
        desired_heading = np.arctan2(dy, dx)
        dtheta = desired_heading - pose[2]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        # Normalize relative goal (scale by arena size)
        dx_norm = dx / self.arena_size
        dy_norm = dy / self.arena_size
        dtheta_norm = dtheta / np.pi  # Normalize to [-1, 1]
        
        # Normalize distance to goal (by max possible distance in arena)
        # Max distance in 10x10 arena from corner to corner = sqrt(200) ≈ 14.14m
        max_distance = self.arena_size * np.sqrt(2)
        distance_norm = distance_to_goal / max_distance
        
        # Velocities
        v = self.robot.last_v
        w = self.robot.last_w
        
        # Normalize velocities
        v_norm = v / 0.7  # Max linear velocity
        w_norm = w / 2.0  # Max angular velocity
        
        # Lidar scan
        lidar = self.robot.get_lidar_scan()
        min_lidar = np.min(lidar)
        
        # Normalize lidar (by max range 5.0m)
        lidar_norm = lidar / 5.0
        min_lidar_norm = min_lidar / 5.0
        
        # REMOVED: Observation logging - too verbose during training
        # Only log during evaluation if needed (controlled by evaluation script)
        
        # Combine observation - NOW INCLUDES DISTANCE
        obs = np.concatenate([
            [dx_norm, dy_norm, dtheta_norm, distance_norm, v_norm, w_norm],
            lidar_norm,
            [min_lidar_norm]
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Additional info for debugging."""
        pose = self.robot.get_pose()
        dist = np.sqrt((self.goal[0]-pose[0])**2 + (self.goal[1]-pose[1])**2)
        return {
            "distance_to_goal": dist,
            "steps": self.step_count,
            "collision": self.robot.check_collision() if self.robot else False
        }
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self._create_environment()
        self.step_count = 0
        self.recent_progress = []  # Reset progress tracking
        
        # Initial distance for reward shaping
        pose = self.robot.get_pose()
        self.prev_distance = np.sqrt(
            (self.goal[0]-pose[0])**2 + (self.goal[1]-pose[1])**2
        )
        
        # #region agent log
        try:
            with open('/Users/abhi/Desktop/Github_Projects/pybullet_sim/.cursor/debug.log', 'a') as f:
                log_entry = json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                    "location": "pybullet_nav_env.py:reset",
                    "message": "Environment reset",
                    "data": {
                        "start_pos": [float(pose[0]), float(pose[1]), float(pose[2])],
                        "goal_pos": [float(self.goal[0]), float(self.goal[1])],
                        "initial_distance": float(self.prev_distance),
                        "goal_threshold": float(self.goal_threshold)
                    },
                    "timestamp": int(os.times().elapsed * 1000)
                }) + "\n"
                f.write(log_entry)
        except: pass
        # #endregion
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action."""
        # Store original action for reward calculation (before clipping)
        original_w = action[1]
        
        # Apply action (clamp to action space) - forward-only
        v, w = action[0], action[1]
        v = np.clip(v, 0.0, 0.7)  # No backward driving
        w = np.clip(w, -2.0, 2.0)
        
        self.robot.set_velocity(v, w)
        
        # Step physics
        num_physics_steps = int(self.time_step / (1/240))
        for _ in range(num_physics_steps):
            p.stepSimulation()
        
        self.step_count += 1
        
        # Get state
        obs = self._get_obs()
        info = self._get_info()
        
        # Compute reward - Simple progress-based approach (based on IR-SIM)
        current_distance = info["distance_to_goal"]
        progress = self.prev_distance - current_distance
        self.prev_distance = current_distance
        
        # Track recent progress to detect stuck behavior
        self.recent_progress.append(progress)
        if len(self.recent_progress) > 20:  # Keep last 20 steps
            self.recent_progress.pop(0)
        
        reward = 0.0
        reward_components = {}
        
        # SIMPLIFIED REWARD FUNCTION - Based on best practices
        # Focus: Progress, No collision, Efficiency, No backward driving
        
        # 1. PRIMARY: Progress toward goal (distance reduction)
        # Scale: 1.0 per meter of progress (simple and effective)
        progress_reward = progress  # progress is already (prev_distance - current_distance)
        reward += progress_reward
        reward_components["progress"] = float(progress_reward)
        
        # 1b. HEADING ALIGNMENT: Small reward for facing towards goal
        # This helps especially when far from goal - encourages proper orientation
        # Get heading error from observation (dtheta_norm is in obs[2])
        dtheta_norm = obs[2]  # Normalized heading error [-1, 1]
        dtheta_abs = abs(dtheta_norm)  # Absolute heading error [0, 1]
        
        # Reward for good heading alignment (smaller heading error = better)
        # Only apply when far from goal (when heading matters more)
        # Scale: 0.1 reward when perfectly aligned, decreases with error
        heading_reward = 0.0
        if current_distance > 1.0:  # Only when far from goal
            heading_reward = 0.1 * (1.0 - dtheta_abs)  # Max 0.1 when aligned, 0 when 90+ deg off
            reward += heading_reward
        reward_components["heading_reward"] = float(heading_reward)
        
        # 2. GOAL REACHED: Large bonus when goal is reached
        goal_bonus = 0.0
        if current_distance < self.goal_threshold:
            goal_bonus = 50.0  # Large but not overwhelming bonus
            reward += goal_bonus
        reward_components["goal_bonus"] = float(goal_bonus)
        
        # 3. COLLISION: Large penalty for collisions
        collision_penalty = 0.0
        if info["collision"]:
            collision_penalty = -10.0  # Significant penalty
            reward += collision_penalty
        reward_components["collision_penalty"] = float(collision_penalty)
        
        # 4. TIME: Small penalty to encourage efficiency (shortest path)
        time_penalty = -0.01
        reward += time_penalty
        reward_components["time_penalty"] = float(time_penalty)
        
        # 5. STUCK: Only penalize if truly stuck (not making progress for many steps)
        # Reduced sensitivity - only trigger if no progress for 20+ steps
        stuck_penalty = 0.0
        is_stuck = False
        if len(self.recent_progress) >= 20:  # Check last 20 steps (was 10)
            recent_total_progress = sum(self.recent_progress[-20:])  # Last 20 steps
            # Only consider stuck if: no progress AND far from goal
            if abs(recent_total_progress) < self.stuck_threshold and current_distance > 1.0:
                stuck_penalty = -0.1  # Small penalty (was -0.5, too harsh)
                reward += stuck_penalty
                is_stuck = True
        reward_components["stuck_penalty"] = float(stuck_penalty)
        
        # Note: Removed complex bonuses/penalties that were causing confusion
        # - Removed close_bonus, very_close_bonus (too complex)
        # - Removed away_penalty (progress reward already handles this)
        # - Removed zero_velocity_penalty (let progress reward handle it)
        # - Simplified stuck detection (less sensitive)
        
        # #region agent log - stuck detection (only log when stuck, not every step)
        # Only log stuck detection - this is rare and important for debugging
        if is_stuck and self.step_count % 10 == 0:  # Only log every 10th stuck detection to reduce volume
            try:
                pose = self.robot.get_pose()
                lidar = self.robot.get_lidar_scan()
                min_lidar = np.min(lidar)
                
                # Check if goal is reachable (raycast from robot to goal)
                robot_pos_3d = [pose[0], pose[1], 0.2]
                goal_pos_3d = [self.goal[0], self.goal[1], 0.2]
                ray_result = p.rayTest(robot_pos_3d, goal_pos_3d)
                goal_visible = True
                hit_fraction = 1.0
                if ray_result and ray_result[0][0] != -1 and ray_result[0][0] != 0:
                    hit_fraction = ray_result[0][2]
                    goal_visible = (hit_fraction > 0.95)  # Goal is visible if ray doesn't hit obstacle
                
                with open('/Users/abhi/Desktop/Github_Projects/pybullet_sim/.cursor/debug.log', 'a') as f:
                    log_entry = json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "pybullet_nav_env.py:step",
                        "message": "Robot stuck detected",
                        "data": {
                            "step": int(self.step_count),
                            "robot_pos": [float(pose[0]), float(pose[1]), float(pose[2])],
                            "goal_pos": [float(self.goal[0]), float(self.goal[1])],
                            "current_distance": float(current_distance),
                            "recent_total_progress": float(recent_total_progress),
                            "min_lidar": float(min_lidar),
                            "collision": bool(self.robot.check_collision() if self.robot else False),
                            "goal_visible": bool(goal_visible),
                            "ray_hit_fraction": float(hit_fraction),
                            "action": [float(v), float(w)]
                        },
                        "timestamp": int(os.times().elapsed * 1000)
                    }) + "\n"
                    f.write(log_entry)
            except: pass
        # #endregion
        
        # Obstacle proximity penalty (based on lidar) - helps avoid collisions
        min_lidar = np.min(obs[6:42])  # Lidar readings in observation (shifted by 1 for distance)
        obstacle_penalty = 0.0
        if min_lidar < 0.3:  # Close to obstacle
            obstacle_penalty = -1.0 * (0.3 - min_lidar)  # Reduced from -2.0 (was too harsh)
            reward += obstacle_penalty
        reward_components["obstacle_penalty"] = float(obstacle_penalty)
        
        # Penalty for excessive rotation (prevents spinning in place)
        # Only penalize if angular velocity is very high (> 2.0 rad/s) - less sensitive
        abs_angular_vel = abs(original_w)  # Use original action before clipping
        rotation_penalty = 0.0
        if abs_angular_vel > 2.0:  # Increased threshold from 1.5 to 2.0
            rotation_penalty = -0.2 * (abs_angular_vel - 2.0)  # Reduced from -0.5
            reward += rotation_penalty
        reward_components["rotation_penalty"] = float(rotation_penalty)
        
        # REMOVED: Reward computation logging - too verbose during training
        # Only log during evaluation if needed (controlled by evaluation script)
        
        # Termination conditions (ensure Python bool, not numpy bool)
        terminated = bool(current_distance < self.goal_threshold)
        truncated = bool(self.step_count >= self.max_steps)
        
        # Truncate on collision (severe collision)
        if info["collision"]:
            truncated = True
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render (handled by PyBullet GUI)."""
        if self.render_mode == "human":
            # Camera follows robot
            try:
                pose = self.robot.get_pose()
                p.resetDebugVisualizerCamera(
                    cameraDistance=8,
                    cameraYaw=np.degrees(pose[2]) + 45,
                    cameraPitch=-45,
                    cameraTargetPosition=[pose[0], pose[1], 0]
                )
            except:
                pass
    
    def close(self):
        """Cleanup."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

