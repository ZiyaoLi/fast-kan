import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)

    def forward(self, x):
        return torch.exp(-(x[..., None] - self.grid) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        base_activation = nn.SiLU,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.base_activation = base_activation()
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(output_dim, input_dim, num_grids)
        )
        nn.init.trunc_normal_(self.spline_weight, mean=0.0, std=spline_weight_init_scale)

    def forward(self, x):
        base = self.base_linear(self.base_activation(x))
        spline_basis = self.rbf(self.layernorm(x))
        spline = torch.einsum(
            "...in,oin->...o", spline_basis, self.spline_weight
        )
        return base + spline


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        base_activation = nn.SiLU,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
