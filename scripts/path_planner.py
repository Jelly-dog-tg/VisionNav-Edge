#!/usr/bin/env python3
"""Path planning algorithms for robot navigation."""
import numpy as np
import heapq
from typing import List, Tuple, Optional, Callable
import math

class AStarPlanner:
    """A* path planning algorithm for grid-based environments."""
    
    def __init__(self, grid_size=0.2, obstacle_threshold=0.5):
        """
        Initialize A* planner.
        
        Args:
            grid_size: Size of each grid cell in meters
            obstacle_threshold: Distance threshold for obstacles
        """
        self.grid_size = grid_size
        self.obstacle_threshold = obstacle_threshold
        
    def create_grid_from_lidar(self, robot_pose: np.ndarray, lidar_scan: np.ndarray, 
                              map_size: Tuple[float, float] = (20.0, 20.0)) -> np.ndarray:
        """
        Create occupancy grid from lidar scan.
        
        Args:
            robot_pose: [x, y, theta] robot position
            lidar_scan: Array of lidar distances
            map_size: (width, height) of map in meters
            
        Returns:
            2D occupancy grid (0=free, 1=obstacle)
        """
        width_cells = int(map_size[0] / self.grid_size)
        height_cells = int(map_size[1] / self.grid_size)
        grid = np.zeros((height_cells, width_cells), dtype=np.uint8)
        
        # Origin in grid (robot position)
        origin_x = width_cells // 2
        origin_y = height_cells // 2
        
        num_rays = len(lidar_scan)
        for i, dist in enumerate(lidar_scan):
            if dist < self.obstacle_threshold:
                # Ray angle
                angle = robot_pose[2] + (2 * np.pi * i / num_rays)
                
                # Obstacle position
                obs_x = robot_pose[0] + dist * np.cos(angle)
                obs_y = robot_pose[1] + dist * np.sin(angle)
                
                # Convert to grid coordinates
                grid_x = int(origin_x + obs_x / self.grid_size)
                grid_y = int(origin_y + obs_y / self.grid_size)
                
                # Mark obstacle and surrounding cells
                if 0 <= grid_x < width_cells and 0 <= grid_y < height_cells:
                    # Mark obstacle cell
                    grid[grid_y, grid_x] = 1
                    # Mark nearby cells as obstacles (safety margin)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = grid_x + dx, grid_y + dy
                            if 0 <= nx < width_cells and 0 <= ny < height_cells:
                                grid[ny, nx] = 1
        
        return grid, (origin_x, origin_y)
    
    def world_to_grid(self, world_pos: Tuple[float, float], origin: Tuple[int, int]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(origin[0] + world_pos[0] / self.grid_size)
        grid_y = int(origin[1] + world_pos[1] / self.grid_size)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos: Tuple[int, int], origin: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        world_x = (grid_pos[0] - origin[0]) * self.grid_size
        world_y = (grid_pos[1] - origin[1]) * self.grid_size
        return (world_x, world_y)
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             grid: np.ndarray, origin: Tuple[int, int]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using A* algorithm.
        
        Args:
            start: (x, y) start position in world coordinates
            goal: (x, y) goal position in world coordinates
            grid: Occupancy grid
            origin: Grid origin in grid coordinates
            
        Returns:
            List of waypoints in world coordinates, or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start, origin)
        goal_grid = self.world_to_grid(goal, origin)
        
        # Check bounds
        height, width = grid.shape
        if (not (0 <= start_grid[0] < width and 0 <= start_grid[1] < height) or
            not (0 <= goal_grid[0] < width and 0 <= goal_grid[1] < height)):
            return None
        
        # Check if start or goal is in obstacle
        if grid[start_grid[1], start_grid[0]] == 1:
            return None
        if grid[goal_grid[1], goal_grid[0]] == 1:
            return None
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        # 8-connected neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(current, origin))
                    current = came_from[current]
                path.append(start)  # Add start
                path.reverse()
                return path
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                
                # Check obstacle
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue
                
                # Cost (diagonal moves cost more)
                move_cost = np.sqrt(dx**2 + dy**2)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found


class RRTPlanner:
    """Rapidly-exploring Random Tree (RRT) path planner for continuous space."""
    
    def __init__(self, step_size=0.5, max_iterations=5000, goal_threshold=0.5):
        """
        Initialize RRT planner.
        
        Args:
            step_size: Maximum step size for tree expansion
            max_iterations: Maximum iterations to find path
            goal_threshold: Distance threshold to consider goal reached
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two points."""
        return np.linalg.norm(a - b)
    
    def is_collision_free(self, start: np.ndarray, end: np.ndarray, 
                         obstacle_checker: Callable) -> bool:
        """
        Check if path segment is collision-free.
        
        Args:
            start: Start point [x, y]
            end: End point [x, y]
            obstacle_checker: Function that takes (x, y) and returns True if obstacle
        """
        # Sample points along the segment
        num_samples = max(5, int(self.distance(start, end) / 0.1))
        for i in range(num_samples + 1):
            alpha = i / num_samples
            point = start + alpha * (end - start)
            if obstacle_checker(point[0], point[1]):
                return False
        return True
    
    def nearest_node(self, tree: List[Tuple[np.ndarray, int]], point: np.ndarray) -> int:
        """Find nearest node in tree to given point."""
        min_dist = float('inf')
        nearest_idx = 0
        for idx, (node, _) in enumerate(tree):
            dist = self.distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx
    
    def steer(self, from_node: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Steer from node toward point with step size limit."""
        direction = to_point - from_node
        dist = self.distance(from_node, to_point)
        if dist <= self.step_size:
            return to_point
        return from_node + (direction / dist) * self.step_size
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             bounds: Tuple[Tuple[float, float], Tuple[float, float]],
             obstacle_checker: Callable) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using RRT algorithm.
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            bounds: ((x_min, x_max), (y_min, y_max)) workspace bounds
            obstacle_checker: Function(x, y) -> bool (True if obstacle)
            
        Returns:
            List of waypoints, or None if no path found
        """
        start = np.array(start)
        goal = np.array(goal)
        
        # Initialize tree
        tree = [(start, -1)]  # (node, parent_index)
        
        for iteration in range(self.max_iterations):
            # Sample random point (with bias toward goal)
            if np.random.random() < 0.1:  # 10% bias toward goal
                sample = goal
            else:
                x_min, x_max = bounds[0]
                y_min, y_max = bounds[1]
                sample = np.array([np.random.uniform(x_min, x_max),
                                  np.random.uniform(y_min, y_max)])
            
            # Find nearest node
            nearest_idx = self.nearest_node(tree, sample)
            nearest_node = tree[nearest_idx][0]
            
            # Steer toward sample
            new_node = self.steer(nearest_node, sample)
            
            # Check collision
            if self.is_collision_free(nearest_node, new_node, obstacle_checker):
                tree.append((new_node, nearest_idx))
                
                # Check if goal reached
                if self.distance(new_node, goal) < self.goal_threshold:
                    # Reconstruct path
                    path = [goal]
                    idx = len(tree) - 1
                    while idx >= 0:
                        node, parent_idx = tree[idx]
                        path.append(node)
                        idx = parent_idx
                    path.append(start)
                    path.reverse()
                    return path
        
        return None  # No path found


class PathFollower:
    """Follow a planned path using PID control."""
    
    def __init__(self, lookahead_distance=0.5, waypoint_threshold=0.3):
        """
        Initialize path follower.
        
        Args:
            lookahead_distance: Distance ahead to look for target point
            waypoint_threshold: Distance to consider waypoint reached
        """
        self.lookahead_distance = lookahead_distance
        self.waypoint_threshold = waypoint_threshold
        self.current_path = None
        self.current_waypoint_idx = 0
        self.last_heading_error = 0.0
        
        # PID gains
        self.kp_linear = 1.2  # Increased for faster movement
        self.kp_angular = 2.0
        self.kd_angular = 0.5
        self.max_linear_vel = 0.7  # Increased from 0.5
        self.max_angular_vel = 2.0
    
    def set_path(self, path: List[Tuple[float, float]]):
        """Set path to follow."""
        self.current_path = path
        self.current_waypoint_idx = 0
    
    def get_target_point(self, robot_pose: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Get target point on path based on lookahead distance.
        
        Args:
            robot_pose: [x, y, theta] current robot pose
            
        Returns:
            (x, y) target point, or None if path complete
        """
        if self.current_path is None or len(self.current_path) == 0:
            return None
        
        robot_pos = robot_pose[:2]
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = self.current_waypoint_idx
        
        for i in range(self.current_waypoint_idx, len(self.current_path)):
            dist = np.linalg.norm(robot_pos - np.array(self.current_path[i]))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look ahead along path
        target_idx = closest_idx
        for i in range(closest_idx, len(self.current_path)):
            dist = np.linalg.norm(robot_pos - np.array(self.current_path[i]))
            if dist >= self.lookahead_distance:
                target_idx = i
                break
            target_idx = i
        
        # Check if waypoint reached
        if (self.current_waypoint_idx < len(self.current_path) and
            np.linalg.norm(robot_pos - np.array(self.current_path[self.current_waypoint_idx])) < self.waypoint_threshold):
            self.current_waypoint_idx += 1
        
        if target_idx >= len(self.current_path):
            return None
        
        return self.current_path[target_idx]
    
    def compute_control(self, robot_pose: np.ndarray) -> Tuple[float, float, bool]:
        """
        Compute control commands to follow path.
        
        Args:
            robot_pose: [x, y, theta] current robot pose
            
        Returns:
            (v, w, path_complete) velocity commands and completion flag
        """
        target = self.get_target_point(robot_pose)
        
        if target is None:
            return 0.0, 0.0, True  # Path complete
        
        # Calculate heading to target
        dx = target[0] - robot_pose[0]
        dy = target[1] - robot_pose[1]
        desired_heading = np.arctan2(dy, dx)
        
        heading_error = desired_heading - robot_pose[2]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # PID control
        p_term = self.kp_angular * heading_error
        d_error = heading_error - self.last_heading_error
        d_term = self.kd_angular * d_error
        
        w = p_term + d_term
        w = np.clip(w, -self.max_angular_vel, self.max_angular_vel)
        self.last_heading_error = heading_error
        
        # Linear velocity
        distance_to_target = np.sqrt(dx**2 + dy**2)
        v = self.kp_linear * distance_to_target
        v = min(v, self.max_linear_vel)
        
        # Reduce speed when turning
        if abs(heading_error) > np.pi/4:
            v *= 0.6
        
        return v, w, False

