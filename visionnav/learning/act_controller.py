"""P8.4: Closed-loop ACT chunk controller for the existing PyBullet navigation env.

Only consumes online 40D observations; the trained CVAE posterior is NOT used.
Expert action units are converted to the normalized action interface used by
RobotNavigationEnv. No BC/PPO/Pure-Pursuit fallback is added.
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from visionnav.learning.act_model import ACTConfig, ACTPolicy
from visionnav.learning.rl_navigation_env import MAX_LINEAR_V, MAX_ANGULAR_W


class ACTChunkController:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )
        self.config = ACTConfig(**checkpoint["model_config"])
        self.model = ACTPolicy(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

        normalization = checkpoint["normalization"]
        self.obs_mean = np.asarray(normalization["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(normalization["obs_std"], dtype=np.float32)
        self.action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(normalization["action_std"], dtype=np.float32)
        if self.obs_mean.shape != (40,) or self.obs_std.shape != (40,):
            raise ValueError("Invalid observation normalization in checkpoint")
        if self.action_mean.shape != (2,) or self.action_std.shape != (2,):
            raise ValueError("Invalid action normalization in checkpoint")
        if (self.obs_std <= 0).any() or (self.action_std <= 0).any():
            raise ValueError("Normalization std must be positive")
        self.history_len = self.config.history_len
        self.execution_steps = int(checkpoint.get("execution_steps", 4))
        if not 1 <= self.execution_steps <= self.config.chunk_size:
            raise ValueError("execution_steps must be within chunk_size")
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.history_len)
        self.pending = None
        self.t = 0
        self.replans = 0
        self.clipped_commands = 0

    @torch.inference_mode()
    def next_action(self, observation):
        """Return env-normalized [v,w], plus a physical-unit diagnostic dict."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.config.obs_dim,) or not np.isfinite(obs).all():
            raise ValueError("ACT requires finite 40D live observations")
        self.history.append(obs.copy())
        if self.pending is None or self.t % self.execution_steps == 0:
            # Initial history padding repeats the first observed state. Record
            # this warm-start in diagnostics; never borrow future states.
            items = list(self.history)
            padded = [items[0]] * (self.history_len - len(items)) + items
            history = np.stack(padded, axis=0)
            normalized = (history - self.obs_mean) / self.obs_std
            inputs = torch.from_numpy(
                np.ascontiguousarray(normalized[None], dtype=np.float32)
            ).to(self.device)
            chunk, _, _ = self.model(inputs)  # deterministic CVAE prior z=0
            raw_chunk = chunk[0].detach().cpu().numpy()
            self.pending = raw_chunk * self.action_std + self.action_mean
            if not np.isfinite(self.pending).all():
                raise RuntimeError("ACT predicted invalid physical commands")
            self.replans += 1

        chunk_index = self.t % self.execution_steps
        raw = self.pending[chunk_index].astype(np.float32)
        physical = np.array([
            np.clip(raw[0], 0.0, MAX_LINEAR_V),
            np.clip(raw[1], -MAX_ANGULAR_W, MAX_ANGULAR_W),
        ], dtype=np.float32)
        was_clipped = not np.allclose(raw, physical, atol=1e-7, rtol=0)
        self.clipped_commands += int(was_clipped)
        normalized_action = np.array([
            2.0 * physical[0] / MAX_LINEAR_V - 1.0,
            physical[1] / MAX_ANGULAR_W,
        ], dtype=np.float32)
        self.t += 1
        return normalized_action, {
            "v_mps": float(physical[0]),
            "w_radps": float(physical[1]),
            "v_raw_mps": float(raw[0]),
            "w_raw_radps": float(raw[1]),
            "action_clipped": bool(was_clipped),
            "history_frames": len(self.history),
            "chunk_index": int(chunk_index),
            "replans": int(self.replans),
        }
