import torch.nn as nn
from cm_kan.ml.layers.cm_kan.memory_efficient_kan import MemoryEfficientHyperspectralCmKANLayer
from cm_kan.core import Logger


class MinimalHyperspectralCmKAN(nn.Module):
    """
    Ultra-minimal hyperspectral CM-KAN for memory-constrained environments
    """
    
    def __init__(
        self,
        in_dims=[31],
        out_dims=[31], 
        grid_size=2,
        spline_order=1,
        residual_std=0.1,
        grid_range=(-1.0, 1.0),
        chunk_size=512
    ):
        super().__init__()
        
        Logger.info(f"MinimalHyperspectralCmKAN: in_dims={in_dims}, out_dims={out_dims}")
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        
        self.kan_layer = MemoryEfficientHyperspectralCmKANLayer(
            in_channels=in_dims[0],
            out_channels=out_dims[0],
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
            chunk_size=chunk_size
        )
        
        self.input_norm = nn.BatchNorm2d(in_dims[0])
        self.output_norm = nn.BatchNorm2d(out_dims[0])
        
    def forward(self, x):
        """Ultra-simple forward pass"""
        # Input normalization
        x = self.input_norm(x)
        
        # Single KAN layer
        x = self.kan_layer(x)
        
        # Output normalization
        x = self.output_norm(x)
        
        return x
