
"""Semantic targets for the Warehouse environment."""

import pybullet as p


# Original data-split layouts plus an independent relocation acceptance layout.
LAYOUTS = {
    "train": [
        (2.00, 0.00),
        (2.35, 0.75),
        (2.30, -0.70),
    ],
    "val": [
        (2.20, 0.18),
        (2.60, 0.90),
        (2.30, -0.85),
    ],
    "test": [
        (2.50, 0.00),
        (2.20, 0.70),
        (2.00, -0.80),
    ],
    "relocated": [
        (-1.40, 1.00),
        (3.30, 1.40),
        (-1.10, -1.25),
    ],
}


def add_semantic_targets(layout="train"):
    """Add box, cylinder and sphere with stable instance IDs."""

    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout}")

    specs = [
        ("box_red_01", 1, "box", "red",
         [0.9, 0.15, 0.15, 1.0]),

        ("cylinder_blue_01", 2, "cylinder", "blue",
         [0.15, 0.3, 0.9, 1.0]),

        ("sphere_green_01", 3, "sphere", "green",
         [0.2, 0.8, 0.2, 1.0]),
    ]

    targets = []

    for spec, (x, y) in zip(specs, LAYOUTS[layout]):
        instance_id, mask_id, cls, color, rgba = spec

        if cls == "box":
            shape = p.GEOM_BOX
            params = {"halfExtents": [0.18] * 3}
            visual_params = params

        elif cls == "cylinder":
            shape = p.GEOM_CYLINDER
            params = {"radius": 0.18, "height": 0.36}
            visual_params = {"radius": 0.18, "length": 0.36}

        else:
            shape = p.GEOM_SPHERE
            params = {"radius": 0.18}
            visual_params = params

        collision = p.createCollisionShape(
            shape, **params
        )

        visual = p.createVisualShape(
            shape,
            rgbaColor=rgba,
            **visual_params,
        )

        position = [float(x), float(y), 0.18]

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
        )

        targets.append({
            "instance_id": instance_id,
            "mask_id": mask_id,
            "class": cls,
            "color": color,
            "position": position,
            "body_id": int(body_id),
        })

    return targets

def add_static_detour_obstacle():
    """Add a low barrier that blocks the direct driving path."""

    half_extents = [0.18, 0.32, 0.075]
    position = [1.05, 0.0, 0.075]

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.35, 0.35, 0.35, 1.0],
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


# Scenario geometry only. All scenarios reuse add_semantic_targets("test").
# These screens do not supply semantic target positions to the search layer.
OCCLUSION_SCREENS = {
    "occluded-blue": ([1.10, 0.35, 0.50], [0.16, 0.55, 0.50]),
    "occluded-green": ([1.00, -0.40, 0.50], [0.16, 0.50, 0.50]),
    "occluded-red": ([1.10, 0.00, 0.50], [0.16, 0.40, 0.50]),
}


def add_occlusion_obstacle(scenario):
    """Add one scenario-specific opaque screen with a real collision body."""
    if scenario not in OCCLUSION_SCREENS:
        raise ValueError(f"Unknown occlusion scenario: {scenario}")
    position, half_extents = OCCLUSION_SCREENS[scenario]
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                 rgbaColor=[0.45, 0.45, 0.45, 1.0])
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                             baseVisualShapeIndex=visual, basePosition=position)


def add_occluded_blue_obstacle():
    """Compatibility entry point for the original blue scenario."""
    return add_occlusion_obstacle("occluded-blue")
