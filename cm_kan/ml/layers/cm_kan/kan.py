from typing import List

import torch
import torch.nn.functional as F

from .bspline import compute_bspline


def generate_control_points(
    low_bound: float,
    up_bound: float,
    in_dim: int,
    spline_order: int,
    grid_size: int):
    spacing = (up_bound - low_bound) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1) * spacing + low_bound
    
    return grid.unsqueeze(0).expand(in_dim, -1).contiguous()


class KANActivation:
    """
    Defines a KAN Activation layer that computes the spline(x) logic
    described in the original paper.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spline_order: int,
        grid_size: int,
        grid_range: List[float],
    ):
        super(KANActivation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.grid_range = grid_range

        self.coef_shape = (out_dim, in_dim, grid_size + spline_order)

        self.grid = generate_control_points(
            grid_range[0],
            grid_range[1],
            in_dim,
            # out_dim,
            spline_order,
            grid_size,
        )

        self.univarate_fn = compute_bspline

    def __call__(self, x: torch.Tensor, coef) -> torch.Tensor:
        """
        Compute and evaluate the learnable activation functions
        applied to a batch of inputs of size in_dim each.
        """
        grid  = self.grid.to(x.device)
        bases = self.univarate_fn(x, grid, self.spline_order)   # (B, in, M)
        spline = torch.einsum('bim,oim->boi', bases, coef)      # (B, out, in)
        return spline


class WeightedResidualLayer:
    """
    Defines the activation function used in the paper,
    phi(x) = w_b SiLU(x) + w_s B_spline(x)
    as a layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        residual_std: float = 0.1,
    ):
        super(WeightedResidualLayer, self).__init__()
        # Residual activation functions
        self.residual_fn = F.silu
        self.univariate_weight_shape = (out_dim, in_dim)
        self.residual_weight_shape = (out_dim, in_dim)

    def __call__(
        self,
        x: torch.Tensor,
        post_acts: torch.Tensor,
        univariate_weight,
        residual_weight,
    ) -> torch.Tensor:
        """
        Given the input to a KAN layer and the activation (e.g. spline(x)),
        compute a weighted residual.

        x has shape (bsz, in_dim) and act has shape (bsz, out_dim, in_dim)
        """

        # Broadcast the input along out_dim of post_acts
        res = residual_weight * self.residual_fn(x[:, None, :])
        act = univariate_weight * post_acts
        return res + act


class KANLayer:
    "Defines a KAN layer from in_dim variables to out_dim variables."

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int,
        spline_order: int,
        residual_std: float = 0.1,
        grid_range: List[float] = [-1, 1],
    ):
        super(KANLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Define univariate function (splines in original KAN)
        self.activation_fn = KANActivation(
            in_dim,
            out_dim,
            spline_order,
            grid_size,
            grid_range,
        )

        # Define the residual connection layer used to compute \phi
        self.residual_layer = WeightedResidualLayer(in_dim, out_dim, residual_std)

    def __call__(
        self, x: torch.Tensor, coef, univariate_weight, residual_weight
    ) -> torch.Tensor:
        spline = self.activation_fn(x, coef)
        phi = self.residual_layer(x, spline, univariate_weight, residual_weight)

        out = torch.sum(phi, dim=-1)

        return out
