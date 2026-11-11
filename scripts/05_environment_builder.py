#!/usr/bin/env python3
"""Build realistic environments (warehouse, street) for robot navigation."""
import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EnvironmentBuilder:
    """Build various environment types for robot navigation."""
    
    def __init__(self, physics_client):
        self.physics_client = physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    def create_walls(self, arena_size=10.0, wall_height=0.5, wall_thickness=0.1):
        """Create arena walls."""
        walls = []
        
        # Create 4 walls around the arena
        wall_configs = [
            # (x, y, length, width, height)
            (0, arena_size, arena_size*2, wall_thickness, wall_height),  # Top
            (0, -arena_size, arena_size*2, wall_thickness, wall_height),  # Bottom
            (arena_size, 0, wall_thickness, arena_size*2, wall_height),  # Right
            (-arena_size, 0, wall_thickness, arena_size*2, wall_height),  # Left
        ]
        
        for x, y, length, width, height in wall_configs:
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[length/2, width/2, height/2],
                rgbaColor=[0.6, 0.6, 0.6, 1]
            )
            wall_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length/2, width/2, height/2]
            )
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=[x, y, height/2]
            )
            walls.append(wall_id)
        
        return walls
    
    def create_warehouse_environment(self, arena_size=10.0):
        """Create a warehouse-like environment with shelves and boxes."""
        print("Building warehouse environment...")
        
        # Create walls
        walls = self.create_walls(arena_size)
        
        # Create shelves (rows of boxes)
        obstacles = []
        
        # Shelf rows
        shelf_configs = [
            # (x, y, orientation, length)
            (2, 3, 0, 3),      # Horizontal shelf
            (-2, -3, 0, 3),   # Horizontal shelf
            (3, -2, np.pi/2, 3),  # Vertical shelf
            (-3, 2, np.pi/2, 3),  # Vertical shelf
        ]
        
        for x, y, angle, length in shelf_configs:
            # Create shelf base
            shelf_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[length/2, 0.15, 0.3],
                rgbaColor=[0.4, 0.3, 0.2, 1]  # Brown wood color
            )
            shelf_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length/2, 0.15, 0.3]
            )
            shelf_orn = p.getQuaternionFromEuler([0, 0, angle])
            shelf_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shelf_collision,
                baseVisualShapeIndex=shelf_visual,
                basePosition=[x, y, 0.3],
                baseOrientation=shelf_orn
            )
            obstacles.append(shelf_id)
            
            # Add boxes on shelves
            num_boxes = int(length / 0.5)
            for i in range(num_boxes):
                box_x = x + (i - num_boxes/2 + 0.5) * 0.5 * np.cos(angle)
                box_y = y + (i - num_boxes/2 + 0.5) * 0.5 * np.sin(angle)
                
                box_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.2, 0.2, 0.2],
                    rgbaColor=[0.8, 0.6, 0.4, 1]  # Light brown
                )
                box_collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[0.2, 0.2, 0.2]
                )
                box_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=box_collision,
                    baseVisualShapeIndex=box_visual,
                    basePosition=[box_x, box_y, 0.7],
                    baseOrientation=shelf_orn
                )
                obstacles.append(box_id)
        
        # Standalone obstacles only; the sole red goal is added by semantic_scene.
        standalone_boxes = [
            (-4, -4, 0.3, 0.3, 0.15, [0.5, 0.5, 0.5, 1]),  # Gray pallet
            (4, 4, 0.4, 0.4, 0.2, [0.7, 0.7, 0.2, 1]),     # Yellow box
        ]
        
        for x, y, lx, ly, lz, color in standalone_boxes:
            box_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[lx, ly, lz],
                rgbaColor=color
            )
            box_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[lx, ly, lz]
            )
            box_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=box_collision,
                baseVisualShapeIndex=box_visual,
                basePosition=[x, y, lz]
            )
            obstacles.append(box_id)
        
        print(f"Created warehouse with {len(obstacles)} obstacles")
        return obstacles
    
    def create_street_environment(self, arena_size=10.0):
        """Create a street-like environment with buildings and obstacles."""
        print("Building street environment...")
        
        # Create walls
        walls = self.create_walls(arena_size)
        
        obstacles = []
        
        # Create buildings (larger rectangular obstacles)
        buildings = [
            # (x, y, width, depth, height, color)
            (-6, -6, 1.5, 1.5, 1.0, [0.5, 0.5, 0.5, 1]),   # Gray building
            (6, 6, 1.5, 1.5, 1.0, [0.4, 0.4, 0.4, 1]),    # Dark gray building
            (-6, 6, 1.0, 1.0, 0.8, [0.6, 0.5, 0.4, 1]),    # Brown building
            (6, -6, 1.2, 1.2, 0.9, [0.5, 0.4, 0.3, 1]),   # Dark brown building
        ]
        
        for x, y, width, depth, height, color in buildings:
            building_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[width/2, depth/2, height/2],
                rgbaColor=color
            )
            building_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[width/2, depth/2, height/2]
            )
            building_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=building_collision,
                baseVisualShapeIndex=building_visual,
                basePosition=[x, y, height/2]
            )
            obstacles.append(building_id)
        
        # Create street obstacles (cars, barriers, etc.)
        street_obstacles = [
            # (x, y, type, size, color)
            (2, 0, 'cylinder', (0.3, 0.4), [0.8, 0.2, 0.2, 1]),  # Red car-like
            (-2, 3, 'cylinder', (0.25, 0.35), [0.2, 0.2, 0.8, 1]),  # Blue car
            (0, -3, 'box', (0.4, 0.2, 0.15), [0.9, 0.9, 0.1, 1]),  # Yellow barrier
            (4, -2, 'cylinder', (0.2, 0.3), [0.3, 0.3, 0.3, 1]),  # Gray obstacle
        ]
        
        for x, y, obs_type, size, color in street_obstacles:
            if obs_type == 'cylinder':
                radius, height = size
                visual = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=color
                )
                collision = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height
                )
                pos_z = height / 2
            else:  # box
                lx, ly, lz = size
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[lx, ly, lz],
                    rgbaColor=color
                )
                collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[lx, ly, lz]
                )
                pos_z = lz
            
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, pos_z]
            )
            obstacles.append(obs_id)
        
        # Add some trees/lampposts (thin cylinders)
        for i in range(6):
            angle = 2 * np.pi * i / 6
            radius = 7.0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            tree_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.1,
                length=1.5,
                rgbaColor=[0.4, 0.3, 0.2, 1]  # Brown
            )
            tree_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.1,
                height=1.5
            )
            tree_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=tree_collision,
                baseVisualShapeIndex=tree_visual,
                basePosition=[x, y, 0.75]
            )
            obstacles.append(tree_id)
        
        print(f"Created street environment with {len(obstacles)} obstacles")
        return obstacles
    
    def create_simple_maze(self, arena_size=8.0):
        """Create a simple maze-like environment."""
        print("Building simple maze environment...")
        
        walls = self.create_walls(arena_size)
        obstacles = []
        
        # Create maze walls
        maze_walls = [
            # (x, y, length, width, height, angle)
            (2, 0, 2, 0.1, 0.4, 0),          # Horizontal wall
            (-2, 2, 1.5, 0.1, 0.4, np.pi/2),   # Vertical wall
            (0, -2, 2, 0.1, 0.4, 0),         # Horizontal wall
            (3, -3, 1.5, 0.1, 0.4, np.pi/2), # Vertical wall
        ]
        
        for x, y, length, width, height, angle in maze_walls:
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[length/2, width/2, height/2],
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            wall_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length/2, width/2, height/2]
            )
            wall_orn = p.getQuaternionFromEuler([0, 0, angle])
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=[x, y, height/2],
                baseOrientation=wall_orn
            )
            obstacles.append(wall_id)
        
        print(f"Created maze with {len(obstacles)} obstacles")
        return obstacles

def main():
    """Test environment builder."""
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    
    # Camera setup
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Load ground
    plane_id = p.loadURDF("plane.urdf")
    
    # Create environment builder
    builder = EnvironmentBuilder(physics_client)
    
    # Choose environment type
    env_type = sys.argv[1] if len(sys.argv) > 1 else "warehouse"
    
    if env_type == "warehouse":
        obstacles = builder.create_warehouse_environment()
    elif env_type == "street":
        obstacles = builder.create_street_environment()
    elif env_type == "maze":
        obstacles = builder.create_simple_maze()
    else:
        print(f"Unknown environment type: {env_type}")
        print("Available: warehouse, street, maze")
        obstacles = builder.create_warehouse_environment()
    
    print(f"\nEnvironment created! Close PyBullet window to exit.")
    print("You can now use this environment for navigation testing.")
    
    # Keep simulation running
    import time
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass
    
    p.disconnect()

if __name__ == "__main__":
    main()

