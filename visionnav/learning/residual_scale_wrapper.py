"""PPO V3: fixed normalized residual scaling outside the unchanged V2 env.

Training AND evaluation order:
    BCResidualNavigationEnvV2 -> ResidualScaleWrapper -> Monitor
    -> DummyVecEnv -> VecNormalize

Pass PPO actions directly to this stack. Do not multiply model.predict() output
by the scale again. Physical velocity conversion remains exclusively in V2.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import ArrayLike, NDArray


class ResidualScaleWrapper(gym.ActionWrapper):
    """Clip a normalized 2-D PPO action, then apply one fixed residual scale.

    Action/observation spaces are inherited without replacement. V2's observation,
    reward, terminal flags and original info entries remain unchanged. Diagnostics
    use a collision-free ``v3_residual`` namespace. Physical corrections are read
    from V2's info, never converted a second time in this wrapper.
    """

    def __init__(self, env: gym.Env, residual_scale: float = 0.25) -> None:
        try:
            scale = float(residual_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("residual_scale must be a finite scalar in [0, 1]") from exc
        if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("residual_scale must be a finite scalar in [0, 1]")
        if (not isinstance(env.action_space, spaces.Box)
                or env.action_space.shape != (2,)
                or not np.all(env.action_space.low == -1.0)
                or not np.all(env.action_space.high == 1.0)):
            raise ValueError("Expected a normalized Box(-1, 1) action space of shape (2,)")
        cursor = env
        while isinstance(cursor, gym.Wrapper):
            if isinstance(cursor, ResidualScaleWrapper):
                raise ValueError("ResidualScaleWrapper is already installed; refusing double scaling")
            cursor = cursor.env
        super().__init__(env)
        self._residual_scale = scale
        self.last_action_diagnostics: dict | None = None
        self._action_steps = 0
        self._active_steps = 0

    @property
    def residual_scale(self) -> float:
        """The validated fixed coefficient; no dynamic residual gating."""
        return self._residual_scale

    def action(self, action: ArrayLike) -> NDArray[np.float32]:
        """Validate -> clip to [-1, 1] -> multiply scale -> float32 residual."""
        try:
            if np.iscomplexobj(action):
                raise ValueError("complex actions are not supported")
            # Validate before float32 conversion: large finite values should be
            # clipped, not overflow to Inf during the initial conversion.
            raw = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            self.last_action_diagnostics = {"input_valid": False, "has_nan_inf": False,
                                            "error": "non-real or non-numeric action"}
            raise ValueError("Residual action must be a real numeric vector of shape (2,)") from exc
        self.last_action_diagnostics = {"input_valid": False,
            "has_nan_inf": bool(not np.isfinite(raw).all()), "input_shape": list(raw.shape)}
        if raw.shape != (2,):
            raise ValueError(f"Expected residual action shape (2,), got {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError("Residual action contains NaN or Inf")
        clipped_action = np.clip(raw, -1.0, 1.0)
        scaled_action = clipped_action * self._residual_scale
        scaled_action = scaled_action.astype(np.float32)
        self.last_action_diagnostics = {
            "input_valid": True, "has_nan_inf": False,
            "raw_action": raw.tolist(), "clipped_action": clipped_action.tolist(),
            "scaled_action": scaled_action.tolist(), "residual_scale": self._residual_scale,
            "action_clipped": bool(np.any(raw != clipped_action)),
            # Match the existing V2 ablation's activation definition.
            "residual_active": bool(np.linalg.norm(scaled_action) > 0.05),
            "activation_threshold_l2": 0.05,
        }
        return scaled_action

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        self._action_steps = 0
        self._active_steps = 0
        self.last_action_diagnostics = None
        return result

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        diagnostics = dict(self.last_action_diagnostics)
        self._action_steps += 1
        self._active_steps += int(diagnostics["residual_active"])
        diagnostics["residual_activation_rate"] = self._active_steps / self._action_steps
        # These are the physical residuals V2 actually applied to its BC fusion,
        # before V2's final velocity clipping. No velocity conversion occurs here.
        correction = info.get("residual_correction")
        diagnostics["delta_v_mps"] = float(correction[0]) if correction is not None else None
        diagnostics["delta_w_radps"] = float(correction[1]) if correction is not None else None
        self.last_action_diagnostics = diagnostics
        augmented_info = dict(info)
        key, suffix = "v3_residual", 1
        while key in augmented_info:
            key = f"v3_residual_{suffix}"
            suffix += 1
        augmented_info[key] = dict(diagnostics)
        return observation, reward, terminated, truncated, augmented_info
