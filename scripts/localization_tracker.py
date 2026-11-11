#!/usr/bin/env python3
"""Localization and trajectory tracking for robot navigation."""
import numpy as np
import time
from typing import List, Tuple, Optional
import json
import os


class LocalizationTracker:
    """Track robot position, velocity, and trajectory over time."""
    
    def __init__(self, save_trajectory=True, trajectory_file=None):
        """
        Initialize localization tracker.
        
        Args:
            save_trajectory: Whether to save trajectory to file
            trajectory_file: Path to save trajectory JSON file
        """
        self.save_trajectory = save_trajectory
        self.trajectory_file = trajectory_file or "logs/trajectory.json"
        
        # Trajectory data
        self.poses = []  # List of [x, y, theta, timestamp]
        self.velocities = []  # List of [vx, vy, w, timestamp]
        self.start_time = None
        self.total_distance = 0.0
        self.last_position = None
        
        # Statistics
        self.stats = {
            'total_time': 0.0,
            'total_distance': 0.0,
            'max_velocity': 0.0,
            'avg_velocity': 0.0,
            'num_samples': 0
        }
    
    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self.poses = []
        self.velocities = []
        self.total_distance = 0.0
        self.last_position = None
        self.stats = {
            'total_time': 0.0,
            'total_distance': 0.0,
            'max_velocity': 0.0,
            'avg_velocity': 0.0,
            'num_samples': 0
        }
    
    def update(self, pose: np.ndarray, velocity: Optional[np.ndarray] = None):
        """
        Update tracker with new pose and velocity.
        
        Args:
            pose: [x, y, theta] robot pose
            velocity: [vx, vy, w] robot velocity (optional)
        """
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        timestamp = current_time - self.start_time
        
        # Store pose
        self.poses.append([float(pose[0]), float(pose[1]), float(pose[2]), timestamp])
        
        # Calculate distance traveled
        if self.last_position is not None:
            dx = pose[0] - self.last_position[0]
            dy = pose[1] - self.last_position[1]
            step_distance = np.sqrt(dx**2 + dy**2)
            self.total_distance += step_distance
        
        self.last_position = pose.copy()
        
        # Store velocity if provided
        if velocity is not None:
            v_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
            self.velocities.append([float(velocity[0]), float(velocity[1]), float(velocity[2]), timestamp])
            
            # Update statistics
            if v_magnitude > self.stats['max_velocity']:
                self.stats['max_velocity'] = v_magnitude
        
        self.stats['num_samples'] += 1
    
    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get most recent pose."""
        if len(self.poses) == 0:
            return None
        return np.array(self.poses[-1][:3])
    
    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Get trajectory as list of (x, y, theta) tuples."""
        return [(p[0], p[1], p[2]) for p in self.poses]
    
    def get_distance_traveled(self) -> float:
        """Get total distance traveled."""
        return self.total_distance
    
    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        if self.start_time is None:
            return self.stats
        
        current_time = time.time()
        self.stats['total_time'] = current_time - self.start_time
        self.stats['total_distance'] = self.total_distance
        
        if self.stats['total_time'] > 0:
            self.stats['avg_velocity'] = self.total_distance / self.stats['total_time']
        
        return self.stats.copy()
    
    def save_trajectory_to_file(self):
        """Save trajectory to JSON file."""
        if not self.save_trajectory:
            return
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self.trajectory_file), exist_ok=True)
        
        data = {
            'trajectory': self.poses,
            'velocities': self.velocities,
            'statistics': self.get_statistics()
        }
        
        with open(self.trajectory_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Trajectory saved to {self.trajectory_file}")
    
    def print_statistics(self):
        """Print tracking statistics."""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("LOCALIZATION STATISTICS")
        print("="*50)
        print(f"Total time: {stats['total_time']:.2f} seconds")
        print(f"Total distance: {stats['total_distance']:.2f} meters")
        print(f"Average velocity: {stats['avg_velocity']:.3f} m/s")
        print(f"Max velocity: {stats['max_velocity']:.3f} m/s")
        print(f"Number of samples: {stats['num_samples']}")
        if stats['total_time'] > 0:
            print(f"Sampling rate: {stats['num_samples']/stats['total_time']:.1f} Hz")
        print("="*50)


class PerformanceLogger:
    """Log performance metrics for each processing cycle."""
    
    def __init__(self, log_interval=100):
        """
        Initialize performance logger.
        
        Args:
            log_interval: Log statistics every N cycles
        """
        self.log_interval = log_interval
        self.cycle_count = 0
        
        # Timing data
        self.timings = {
            'sensor_read': [],
            'path_planning': [],
            'navigation_update': [],
            'visualization': [],
            'control_apply': [],
            'physics_step': [],
            'total_cycle': []
        }
        
        self.current_cycle_start = None
    
    def start_cycle(self):
        """Start timing a cycle."""
        self.current_cycle_start = time.time()
        self.cycle_count += 1
    
    def log_timing(self, operation: str, duration: float):
        """
        Log timing for an operation.
        
        Args:
            operation: Name of operation
            duration: Duration in seconds
        """
        if operation in self.timings:
            self.timings[operation].append(duration)
    
    def time_operation(self, operation: str):
        """
        Context manager for timing operations.
        
        Usage:
            with logger.time_operation('sensor_read'):
                # do sensor reading
        """
        return TimingContext(self, operation)
    
    def end_cycle(self):
        """End timing a cycle and log total cycle time."""
        if self.current_cycle_start is not None:
            total_time = time.time() - self.current_cycle_start
            self.timings['total_cycle'].append(total_time)
    
    def get_statistics(self) -> dict:
        """Get performance statistics."""
        stats = {}
        for operation, timings in self.timings.items():
            if len(timings) > 0:
                stats[operation] = {
                    'mean': np.mean(timings),
                    'std': np.std(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'total': np.sum(timings)
                }
            else:
                stats[operation] = None
        return stats
    
    def print_statistics(self):
        """Print performance statistics."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Total cycles: {self.cycle_count}")
        print(f"\nTiming breakdown (milliseconds):")
        print("-" * 60)
        
        for operation, stat in stats.items():
            if stat is not None:
                print(f"{operation:20s}: "
                      f"mean={stat['mean']*1000:6.2f}ms, "
                      f"std={stat['std']*1000:6.2f}ms, "
                      f"min={stat['min']*1000:6.2f}ms, "
                      f"max={stat['max']*1000:6.2f}ms")
        
        # Calculate percentages
        if stats['total_cycle'] is not None:
            total_mean = stats['total_cycle']['mean']
            print(f"\nPercentage of cycle time:")
            print("-" * 60)
            for operation, stat in stats.items():
                if stat is not None and operation != 'total_cycle' and total_mean > 0:
                    percentage = (stat['mean'] / total_mean) * 100
                    print(f"{operation:20s}: {percentage:5.1f}%")
        
        print("="*60)
    
    def should_log(self) -> bool:
        """Check if should log statistics this cycle."""
        return self.cycle_count % self.log_interval == 0


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, logger: PerformanceLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger.log_timing(self.operation, duration)

