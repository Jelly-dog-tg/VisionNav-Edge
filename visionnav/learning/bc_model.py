
"""MLP model for VisionNav-Edge Behavior Cloning."""

import torch
import torch.nn as nn


class BehaviorCloningMLP(nn.Module):
    """
    Behavior Cloning policy.

    Input:
        40-dimensional normalized observation.

    Output:
        2-dimensional normalized expert action.
    """

    def __init__(
        self,
        input_dim: int = 40,
        output_dim: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            # Layer 1: 40 -> 128
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            # Layer 2: 128 -> 128
            nn.Linear(128, 128),
            nn.ReLU(),

            # Layer 3: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output: 64 -> 2
            nn.Linear(64, output_dim),
        )

    def forward(self, observation: torch.Tensor):
        """
        Predict normalized robot actions.
        """

        if observation.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {observation.shape[-1]}"
            )

        return self.network(observation)


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    model = BehaviorCloningMLP()

    # Simulate one observation.
    single_input = torch.randn(40)

    # Simulate a batch of 32 observations.
    batch_input = torch.randn(32, 40)

    single_output = model(single_input)
    batch_output = model(batch_input)

    print("BEHAVIOR CLONING MODEL")
    print("----------------------")

    print("Model architecture:")
    print(model)

    print("\nSingle input shape:", single_input.shape)
    print("Single output shape:", single_output.shape)

    print("\nBatch input shape:", batch_input.shape)
    print("Batch output shape:", batch_output.shape)

    print("\nTrainable parameters:", count_parameters(model))

    print(
        "\nOutput finite:",
        torch.isfinite(batch_output).all().item()
    )

    print("\nMODEL INITIALIZATION PASSED")
