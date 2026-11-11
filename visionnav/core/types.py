
from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class RobotState:
    """机器人状态，位置单位 m，角度单位 rad。"""

    time_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    v_mps: float
    w_radps: float
    frame_id: str = "world"

    def __post_init__(self):
        values = (
            self.time_s, self.x_m, self.y_m,
            self.yaw_rad, self.v_mps, self.w_radps
        )
        if not all(isfinite(v) for v in values):
            raise ValueError("RobotState contains non-finite values")
        if self.time_s < 0:
            raise ValueError("time_s must be non-negative")


@dataclass(frozen=True)
class Action:
    """差速机器人速度指令及控制周期。"""

    v_mps: float
    w_radps: float
    dt_s: float
    source: str = "unknown"

    def __post_init__(self):
        if not all(isfinite(v) for v in (
            self.v_mps, self.w_radps, self.dt_s
        )):
            raise ValueError("Action contains non-finite values")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive")


@dataclass(frozen=True)
class TaskCommand:
    """自然语言任务与结构化目标。"""

    task_id: str
    instruction: str
    target_class: str | None = None
    target_attributes: dict[str, str] = field(
        default_factory=dict
    )
    target_instance_id: int | None = None

    def __post_init__(self):
        if not self.task_id or not self.instruction.strip():
            raise ValueError("Task ID and instruction are required")


@dataclass(frozen=True)
class CameraCalibration:
    """针孔相机参数及相机到世界的变换矩阵。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # 4x4 齐次矩阵，按行展开
    T_world_camera: tuple[float, ...]

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid image size")
        if not all(isfinite(v) for v in (
            self.fx, self.fy, self.cx, self.cy
        )):
            raise ValueError("Invalid camera intrinsics")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")
        if len(self.T_world_camera) != 16:
            raise ValueError("Expected a 4x4 camera transform")
        if not all(isfinite(v) for v in self.T_world_camera):
            raise ValueError("Invalid camera transform")


@dataclass(frozen=True)
class Observation:
    """一次机器人观察；图像以文件引用保存。"""

    timestamp_s: float
    scene_id: str
    task_id: str
    state: RobotState
    lidar_m: tuple[float, ...]
    rgb_path: str | None = None
    depth_path: str | None = None
    segmentation_path: str | None = None
    camera: CameraCalibration | None = None

    def __post_init__(self):
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("Invalid observation timestamp")
        if not self.scene_id or not self.task_id:
            raise ValueError("Scene ID and Task ID are required")
        if not all(isfinite(v) and v >= 0 for v in self.lidar_m):
            raise ValueError("Invalid LiDAR distances")


@dataclass(frozen=True)
class EpisodeStep:
    """单个控制步的观察、动作和结果。"""

    observation: Observation
    action: Action
    reward: float | None = None
    collision: bool = False
    done: bool = False


@dataclass
class Episode:
    """一次完整导航任务。"""

    episode_id: str
    scene_id: str
    task: TaskCommand
    steps: list[EpisodeStep] = field(default_factory=list)
    success: bool | None = None
    termination_reason: str | None = None

    def add_step(self, step: EpisodeStep):
        obs = step.observation

        if obs.scene_id != self.scene_id:
            raise ValueError("Scene mismatch")
        if obs.task_id != self.task.task_id:
            raise ValueError("Task mismatch")
        if self.steps and obs.timestamp_s < (
            self.steps[-1].observation.timestamp_s
        ):
            raise ValueError("Timestamps must be ordered")

        self.steps.append(step)
