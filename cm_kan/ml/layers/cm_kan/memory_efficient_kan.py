import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .kan import KANLayer


class MemoryEfficientBSpline(nn.Module):
    """Memory-efficient B-spline computation using chunking"""
    
    def __init__(self, k=3, grid_size=5, grid_range=(-1.0, 1.0), chunk_size=512):
        super().__init__()
        self.k = k
        self.grid_size = grid_size
        self.chunk_size = chunk_size
        
        # Create grid
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        grid = grid.expand(1, 1, 1, -1)  # Shape: (1, 1, 1, grid_size+1)
        self.register_buffer('grid', grid)
        
    def forward(self, x, coef):
        """Compute B-spline in chunks to save memory"""
        B, out_dim, in_dim, H, W = x.shape[0], coef.shape[1], coef.shape[2], x.shape[2], x.shape[3]
        
        # Reshape for processing
        x_flat = x.view(B, -1, H * W)  # (B, in_dim, H*W)
        
        # Process in chunks to save memory
        total_pixels = H * W
        output_chunks = []
        
        for start_idx in range(0, total_pixels, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_pixels)
            chunk_size_actual = end_idx - start_idx
            
            # Extract chunk
            x_chunk = x_flat[:, :, start_idx:end_idx]  # (B, in_dim, chunk_size)
            
            # Compute B-spline for this chunk
            chunk_result = self._compute_bspline_chunk(x_chunk, coef)
            output_chunks.append(chunk_result)
        
        # Concatenate chunks
        output = torch.cat(output_chunks, dim=-1)  # (B, out_dim, H*W)
        output = output.view(B, out_dim, H, W)
        
        return output
    
    def _compute_bspline_chunk(self, x_chunk, coef):
        """Compute B-spline for a small chunk"""
        B, in_dim, chunk_size = x_chunk.shape
        out_dim = coef.shape[1]
        
        # Simplified B-spline computation for memory efficiency
        # Use linear interpolation instead of full B-spline for extreme memory saving
        
        # Clamp x to grid range
        x_clamped = torch.clamp(x_chunk, self.grid[0, 0, 0, 0], self.grid[0, 0, 0, -1])
        
        # Find grid indices
        grid_expanded = self.grid.expand(B, out_dim, in_dim, -1)
        
        # Simple linear interpolation between grid points
        grid_spacing = (self.grid[0, 0, 0, 1] - self.grid[0, 0, 0, 0])
        normalized_x = (x_clamped - self.grid[0, 0, 0, 0]) / grid_spacing
        
        # Get integer and fractional parts
        grid_idx = torch.floor(normalized_x).long()
        grid_idx = torch.clamp(grid_idx, 0, self.grid_size - 1)
        frac = normalized_x - grid_idx.float()
        
        # Linear interpolation weights
        w1 = 1.0 - frac
        w2 = frac
        
        # Apply coefficients
        result = torch.zeros(B, out_dim, chunk_size, device=x_chunk.device, dtype=x_chunk.dtype)
        
        for b in range(B):
            for o in range(out_dim):
                for i in range(in_dim):
                    idx = grid_idx[b, i, :]
                    idx_next = torch.clamp(idx + 1, 0, self.grid_size)
                    
                    # Linear interpolation with coefficients
                    contrib = (w1[b, i, :] * coef[b, o, i, idx] + 
                              w2[b, i, :] * coef[b, o, i, idx_next])
                    result[b, o, :] += contrib
        
        return result


class MemoryEfficientHyperspectralCmKANLayer(nn.Module):
    """Memory-efficient hyperspectral CM-KAN layer"""
    
    def __init__(
        self,
        in_channels=31,
        out_channels=31,
        grid_size=2,
        spline_order=1,
        residual_std=0.1,
        grid_range=(-1.0, 1.0),
        chunk_size=512
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.chunk_size = chunk_size
        
        # Memory-efficient B-spline
        self.bspline = MemoryEfficientBSpline(
            k=spline_order,
            grid_size=grid_size,
            grid_range=grid_range,
            chunk_size=chunk_size
        )
        
        # Coefficients - this is the main memory consumer
        self.coef = nn.Parameter(
            torch.randn(1, out_channels, in_channels, grid_size + 1) * residual_std
        )
        
        # Simple residual connection
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        """Forward pass with memory efficiency"""
        B, C, H, W = x.shape
        
        # Expand coefficients for batch
        coef_expanded = self.coef.expand(B, -1, -1, -1)
        
        # Apply B-spline transformation
        spline_output = self.bspline(x, coef_expanded)
        
        # Residual connection
        residual_output = self.residual(x)
        
        # Combine (with reduced weight on spline to save memory)
        output = 0.7 * residual_output + 0.3 * spline_output
        
        return output


class ChunkedHyperspectralProcessor(nn.Module):
    """Process hyperspectral data in spatial chunks to save memory"""
    
    def __init__(self, processor_fn, chunk_size=64):
        super().__init__()
        self.processor_fn = processor_fn
        self.chunk_size = chunk_size
    
    def forward(self, x):
        """Process input in spatial chunks"""
        B, C, H, W = x.shape
        
        # If image is small enough, process normally
        if H * W <= self.chunk_size * self.chunk_size:
            return self.processor_fn(x)
        
        # Otherwise, process in chunks
        output_chunks = []
        
        for h_start in range(0, H, self.chunk_size):
            h_end = min(h_start + self.chunk_size, H)
            row_chunks = []
            
            for w_start in range(0, W, self.chunk_size):
                w_end = min(w_start + self.chunk_size, W)
                
                # Extract chunk
                chunk = x[:, :, h_start:h_end, w_start:w_end]
                
                # Process chunk
                processed_chunk = self.processor_fn(chunk)
                row_chunks.append(processed_chunk)
            
            # Concatenate row chunks
            row_output = torch.cat(row_chunks, dim=3)
            output_chunks.append(row_output)
        
        # Concatenate all chunks
        output = torch.cat(output_chunks, dim=2)
        
        return output