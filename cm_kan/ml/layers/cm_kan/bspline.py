import torch

def compute_bspline(x: torch.Tensor, grid: torch.Tensor, k: int):
    # x: (B, in_dim), grid: (in_dim, P)
    grid = grid.unsqueeze(0).to(x.device)          # (1, in_dim, P)
    x    = x.unsqueeze(-1).to(x.device)            # (B, in_dim, 1)
    bases = (x >= grid[..., :-1]) & (x < grid[..., 1:])  # (B, in_dim, P-1)
    bases = bases.to(x.dtype)

    for j in range(1, k + 1):
        n   = grid.size(-1) - (j + 1)
        den = (grid[..., j:-1]   - grid[..., :n]).clamp_min(1e-12)
        b1  = ((x - grid[..., :n])       / den) * bases[..., :-1]
        den = (grid[..., j+1:] - grid[..., 1:n+1]).clamp_min(1e-12)
        b2  = ((grid[..., j+1:] - x) / den)     * bases[..., 1:]
        bases = b1 + b2
    
    return bases  # (B, in_dim, grid_size + k)

    
def coef2curve(
    x: torch.Tensor,
    grid: torch.Tensor,
    coefs: torch.Tensor,
    k: int,
    device: torch.device,
):
    """
    For a given (batch of) x, control points (grid), and B-spline coefficients,
    evaluate and return x on the B-spline function.
    """
    bases = compute_bspline(x, grid, k, device)
    spline = torch.sum(bases * coefs[None, ...], dim=-1)
    return spline


if __name__ == "__main__":
    print("B Spline Unit Tests")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bsz = 2
    spline_order = 3
    in_dim = 5
    out_dim = 7
    grid_size = 11
    grid_range = [-1.0, 1]

    x = torch.ones(bsz, in_dim) / 0.8

    spacing = (grid_range[1] - grid_range[0]) / grid_size
    grid = (
        torch.arange(-spline_order, grid_size + spline_order + 1, device=device)
        * spacing
        + grid_range[0]
    )
    # Generate (out, in) copies of equally spaced points on [a, b]
    grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()

    print("x", x)
    print("grid", grid)

    compute_bspline(x, grid, spline_order, device)
