import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.w = nn.Parameter(
            torch.empty(
                out_features,
                self.in_features,
                device=self.device,
                dtype=self.dtype,
            )
        )

        std = 2 / (self.in_features + self.out_features) ** 0.5
        nn.init.trunc_normal_(self.w, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T
