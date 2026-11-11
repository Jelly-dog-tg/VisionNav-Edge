"""P8.2: Lightweight, state-based ACT-style CVAE Transformer for VisionNav.

This is an ACT-style action-chunking model adapted to 40-D robot state
histories and 2-D differential-drive controls. It is not the image-based
manipulator architecture from the original ACT paper.

Training:  past 8 normalized observations + next 16 normalized expert actions
           -> posterior CVAE -> Transformer decoder -> 16 normalized actions.
Inference: past 8 normalized observations -> z=0 prior mean -> 16 actions.

The trained controller must de-normalize outputs using the saved training-set
statistics and enforce its own physical velocity / collision constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ACTConfig:
    obs_dim: int = 40
    action_dim: int = 2
    history_len: int = 8
    chunk_size: int = 16
    d_model: int = 96
    nhead: int = 4
    obs_layers: int = 2
    posterior_layers: int = 1
    decoder_layers: int = 2
    dim_feedforward: int = 256
    latent_dim: int = 16
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Observation/action dimensions must be positive")
        if self.history_len <= 0 or self.chunk_size <= 0:
            raise ValueError("History and chunk lengths must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")


class ACTPolicy(nn.Module):
    """CVAE + Transformer action-chunk predictor for low-dimensional states."""

    def __init__(self, config: ACTConfig | None = None) -> None:
        super().__init__()
        self.config = config or ACTConfig()
        c = self.config

        self.obs_embed = nn.Linear(c.obs_dim, c.d_model)
        self.obs_pos = nn.Parameter(torch.zeros(1, c.history_len, c.d_model))
        nn.init.normal_(self.obs_pos, std=0.02)

        self.obs_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.nhead,
                dim_feedforward=c.dim_feedforward,
                dropout=c.dropout,
                batch_first=True,
                norm_first=False,
                activation="gelu",
            ),
            num_layers=c.obs_layers,
            enable_nested_tensor=False,
        )

        # Posterior q(z | observation history, future expert actions).
        self.action_embed = nn.Linear(c.action_dim, c.d_model)
        self.action_pos = nn.Parameter(torch.zeros(1, c.chunk_size, c.d_model))
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, c.d_model))
        nn.init.normal_(self.action_pos, std=0.02)
        nn.init.normal_(self.posterior_cls, std=0.02)
        self.posterior_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.nhead,
                dim_feedforward=c.dim_feedforward,
                dropout=c.dropout,
                batch_first=True,
                norm_first=False,
                activation="gelu",
            ),
            num_layers=c.posterior_layers,
            enable_nested_tensor=False,
        )
        self.posterior_head = nn.Linear(c.d_model, 2 * c.latent_dim)

        self.latent_embed = nn.Linear(c.latent_dim, c.d_model)
        self.action_queries = nn.Parameter(
            torch.zeros(1, c.chunk_size, c.d_model)
        )
        nn.init.normal_(self.action_queries, std=0.02)
        self.action_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=c.d_model,
                nhead=c.nhead,
                dim_feedforward=c.dim_feedforward,
                dropout=c.dropout,
                batch_first=True,
                norm_first=False,
                activation="gelu",
            ),
            num_layers=c.decoder_layers,
        )
        self.action_head = nn.Linear(c.d_model, c.action_dim)

    def forward(
        self,
        observations: Tensor,
        future_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Return normalized action chunk and (mu, logvar) during training.

        If future_actions is omitted, use the prior mean z=0 deterministically.
        NEVER provide future expert actions at inference time.
        """
        c = self.config
        if observations.ndim != 3 or observations.shape[1:] != (
            c.history_len,
            c.obs_dim,
        ):
            raise ValueError(
                f"Expected observations [B,{c.history_len},{c.obs_dim}], "
                f"got {tuple(observations.shape)}"
            )

        batch = observations.shape[0]
        memory = self.obs_encoder(
            self.obs_embed(observations) + self.obs_pos
        )
        mu = logvar = None

        if future_actions is not None:
            if future_actions.shape != (batch, c.chunk_size, c.action_dim):
                raise ValueError(
                    f"Expected future actions [B,{c.chunk_size},"
                    f"{c.action_dim}], got {tuple(future_actions.shape)}"
                )
            cls = self.posterior_cls.expand(batch, -1, -1)
            cls = cls + memory.mean(dim=1, keepdim=True)
            future_tokens = (
                self.action_embed(future_actions) + self.action_pos
            )
            posterior_tokens = torch.cat([cls, future_tokens], dim=1)
            latent_hidden = self.posterior_encoder(posterior_tokens)[:, 0]
            mu, logvar = self.posterior_head(latent_hidden).chunk(2, dim=-1)
            logvar = logvar.clamp(min=-10.0, max=10.0)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = observations.new_zeros((batch, c.latent_dim))

        queries = self.action_queries.expand(batch, -1, -1)
        queries = queries + self.latent_embed(z).unsqueeze(1)
        decoded = self.action_decoder(tgt=queries, memory=memory)
        return self.action_head(decoded), mu, logvar


def act_loss(
    predicted: Tensor,
    target: Tensor,
    mu: Tensor | None,
    logvar: Tensor | None,
    kl_weight: float = 1e-3,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Normalized-space L1 + beta * mean-per-latent-dimension KL."""
    if mu is None or logvar is None:
        raise ValueError("CVAE training loss requires posterior parameters")
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target action shapes differ")
    recon = F.l1_loss(predicted, target)
    kl_per_dim = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl = kl_per_dim.mean()
    loss = recon + kl_weight * kl
    return loss, {"reconstruction_l1": recon.detach(), "kl": kl.detach()}


def get_model_config(model: ACTPolicy) -> dict:
    return asdict(model.config)


if __name__ == "__main__":
    # A single model-construction smoke test, not a benchmark.
    torch.manual_seed(42)
    policy = ACTPolicy()
    x = torch.randn(2, 8, 40)
    y = torch.randn(2, 16, 2)
    policy.train()
    pred, mu, logvar = policy(x, y)
    loss, metrics = act_loss(pred, y, mu, logvar)
    loss.backward()
    policy.eval()
    with torch.no_grad():
        output, _, _ = policy(x)
    assert pred.shape == output.shape == (2, 16, 2)
    assert torch.isfinite(output).all()
    print("P8.2 ACT model ready")
    print("Training output:", tuple(pred.shape))
    print("Inference output:", tuple(output.shape))
    print("Trainable parameters:", sum(p.numel() for p in policy.parameters()))
    print("Synthetic forward/backward: PASS")
