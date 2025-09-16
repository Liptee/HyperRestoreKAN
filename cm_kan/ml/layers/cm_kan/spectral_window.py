import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
import numpy as np


class SpectralWindowPreprocessor(nn.Module):
    """
    Spectral window preprocessor for local spectral processing.
    
    Transforms input [B, C, H, W] to [B, C, (2n+1), H, W] where each channel k
    gets a window [k-n, ..., k, ..., k+n] of neighboring spectral channels.
    """
    
    def __init__(
        self,
        num_channels: int = 31,
        window_size: int = 3,
        padding_mode: str = "reflect",
        wavelengths: Optional[List[float]] = None
    ):
        """
        Args:
            num_channels: Number of spectral channels
            window_size: Half-window size (n), total window = 2n+1
            padding_mode: Padding mode for boundary channels ("reflect", "replicate", "zero")
            wavelengths: Optional wavelengths for nm-based window selection
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.window_size = window_size
        self.padding_mode = padding_mode
        self.wavelengths = wavelengths
        self.total_window_size = 2 * window_size + 1
        
        # Validate parameters
        if padding_mode not in ["reflect", "replicate", "zero"]:
            raise ValueError(f"padding_mode must be one of ['reflect', 'replicate', 'zero'], got {padding_mode}")
        
        if wavelengths is not None and len(wavelengths) != num_channels:
            raise ValueError(f"wavelengths length {len(wavelengths)} must match num_channels {num_channels}")
        
        # Pre-compute channel indices for each output channel
        self.register_buffer("channel_indices", self._compute_channel_indices())
    
    def _compute_channel_indices(self) -> torch.Tensor:
        """
        Pre-compute which input channels to use for each output channel's window.
        Returns tensor of shape [num_channels, total_window_size]
        """
        indices = torch.zeros(self.num_channels, self.total_window_size, dtype=torch.long)
        
        for center_ch in range(self.num_channels):
            window_indices = []
            
            for offset in range(-self.window_size, self.window_size + 1):
                target_ch = center_ch + offset
                
                if self.padding_mode == "zero":
                    # Use -1 to indicate zero padding (will be handled in forward)
                    if target_ch < 0 or target_ch >= self.num_channels:
                        window_indices.append(-1)
                    else:
                        window_indices.append(target_ch)
                        
                elif self.padding_mode == "reflect":
                    # Reflect at boundaries
                    if target_ch < 0:
                        target_ch = -target_ch - 1
                    elif target_ch >= self.num_channels:
                        target_ch = 2 * self.num_channels - target_ch - 1
                    window_indices.append(max(0, min(target_ch, self.num_channels - 1)))
                    
                elif self.padding_mode == "replicate":
                    # Clamp to boundaries
                    target_ch = max(0, min(target_ch, self.num_channels - 1))
                    window_indices.append(target_ch)
            
            indices[center_ch] = torch.tensor(window_indices, dtype=torch.long)
        
        return indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral windowing preprocessing.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Windowed tensor of shape [B, C, (2n+1), H, W]
        """
        B, C, H, W = x.shape
        
        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {C}")
        
        # Initialize output tensor
        output = torch.zeros(B, C, self.total_window_size, H, W, 
                           dtype=x.dtype, device=x.device)
        
        # Process each output channel
        for out_ch in range(C):
            window_indices = self.channel_indices[out_ch]
            
            for window_pos in range(self.total_window_size):
                in_ch = window_indices[window_pos].item()
                
                if in_ch == -1:  # Zero padding
                    # output[:, out_ch, window_pos, :, :] already initialized to zero
                    pass
                else:
                    output[:, out_ch, window_pos, :, :] = x[:, in_ch, :, :]
        
        return output
    
    def get_window_for_channel(self, channel_idx: int) -> List[int]:
        """
        Get the window indices for a specific channel (for debugging/visualization).
        
        Args:
            channel_idx: Channel index
            
        Returns:
            List of input channel indices for the window
        """
        if channel_idx >= self.num_channels:
            raise ValueError(f"channel_idx {channel_idx} >= num_channels {self.num_channels}")
        
        return self.channel_indices[channel_idx].tolist()


class GroupedKANLayer(nn.Module):
    """
    Grouped KAN layer for processing windowed spectral data.
    
    Each group processes a (2n+1)-channel window and outputs 1 channel.
    Supports both shared and per-channel parameter generation.
    """
    
    def __init__(
        self,
        num_channels: int = 31,
        window_size: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
        residual_std: float = 0.1,
        grid_range: tuple = (-1.0, 1.0),
        shared_params: bool = True
    ):
        """
        Args:
            num_channels: Number of output channels (groups)
            window_size: Half-window size (n), input window = 2n+1
            grid_size: KAN grid size
            spline_order: Spline order for KAN
            residual_std: Residual connection standard deviation
            grid_range: Range for KAN grid
            shared_params: Whether to share KAN parameters across all groups
        """
        super().__init__()
        
        from .kan import KANLayer
        from .hyperspectral_generator import HyperspectralGeneratorLayer
        
        self.num_channels = num_channels
        self.window_size = window_size
        self.input_dim = 2 * window_size + 1
        self.shared_params = shared_params
        
        # Create template KAN layer to get parameter structure
        self.template_kan = KANLayer(
            in_dim=self.input_dim,
            out_dim=1,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )
        
        # Calculate KAN parameter structure
        self._setup_kan_params()
        
        if shared_params:
            # Single generator for all channels
            self.generator = HyperspectralGeneratorLayer(
                in_channels=num_channels, 
                out_channels=self.kan_params_num
            )
        else:
            # Separate simple generator for each channel
            self.generators = nn.ModuleList([
                self._create_simple_generator() 
                for _ in range(num_channels)
            ])
    
    def _setup_kan_params(self):
        """Setup KAN parameter indices and counts."""
        self.kan_params_indices = [0]
        
        coef_len = np.prod(self.template_kan.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.template_kan.residual_layer.univariate_weight_shape
        )
        residual_weight_len = np.prod(
            self.template_kan.residual_layer.residual_weight_shape
        )
        
        self.kan_params_indices.extend([
            coef_len, 
            univariate_weight_len, 
            residual_weight_len
        ])
        
        self.kan_params_num = sum(self.kan_params_indices)
        self.kan_params_indices = np.cumsum(self.kan_params_indices)
    
    def _create_simple_generator(self):
        """Create a simple parameter generator for individual channels."""
        return nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, self.kan_params_num, 1),
        )
    
    def _apply_kan_single(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Apply KAN transformation to a single sample."""
        # Extract parameters from weights
        i, j = self.kan_params_indices[0], self.kan_params_indices[1]
        coef = weights[i:j].view(*self.template_kan.activation_fn.coef_shape)
        
        i, j = self.kan_params_indices[1], self.kan_params_indices[2]
        univariate_weight = weights[i:j].view(
            *self.template_kan.residual_layer.univariate_weight_shape
        )
        
        i, j = self.kan_params_indices[2], self.kan_params_indices[3]
        residual_weight = weights[i:j].view(
            *self.template_kan.residual_layer.residual_weight_shape
        )
        
        # Apply KAN transformation
        result = self.template_kan(x.unsqueeze(0), coef, univariate_weight, residual_weight)
        return result.squeeze(0)
    
    def forward(self, windowed_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for grouped KAN processing.
        
        Args:
            windowed_x: Input tensor of shape [B, C, (2n+1), H, W]
            
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        B, C, window_dim, H, W = windowed_x.shape
        
        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {C}")
        if window_dim != self.input_dim:
            raise ValueError(f"Expected window dimension {self.input_dim}, got {window_dim}")
        
        # Initialize output
        output = torch.zeros(B, C, H, W, dtype=windowed_x.dtype, device=windowed_x.device)
        
        if self.shared_params:
            # Generate shared parameters using global context
            global_context = windowed_x.mean(dim=2)  # [B, C, H, W]
            weights = self.generator(global_context)  # [B, kan_params_num, H, W]
            
            # Process each channel with shared parameters
            for ch in range(C):
                channel_input = windowed_x[:, ch, :, :, :]  # [B, window_dim, H, W]
                channel_weights = weights  # Shared weights
                
                # Reshape for pixel-wise processing
                channel_input_flat = channel_input.permute(0, 2, 3, 1).reshape(B * H * W, window_dim)
                channel_weights_flat = channel_weights.permute(0, 2, 3, 1).reshape(B * H * W, self.kan_params_num)
                
                # Apply KAN to each pixel
                output_flat = torch.zeros(B * H * W, 1, dtype=windowed_x.dtype, device=windowed_x.device)
                for pixel in range(B * H * W):
                    output_flat[pixel] = self._apply_kan_single(
                        channel_input_flat[pixel], 
                        channel_weights_flat[pixel]
                    )
                
                # Reshape back to spatial format
                output[:, ch, :, :] = output_flat.view(B, H, W)
        
        else:
            # Process each channel with dedicated parameters
            for ch in range(C):
                channel_input = windowed_x[:, ch, :, :, :]  # [B, window_dim, H, W]
                
                # Generate channel-specific parameters
                weights = self.generators[ch](channel_input)  # [B, kan_params_num, H, W]
                
                # Reshape for pixel-wise processing
                channel_input_flat = channel_input.permute(0, 2, 3, 1).reshape(B * H * W, window_dim)
                weights_flat = weights.permute(0, 2, 3, 1).reshape(B * H * W, self.kan_params_num)
                
                # Apply KAN to each pixel
                output_flat = torch.zeros(B * H * W, 1, dtype=windowed_x.dtype, device=windowed_x.device)
                for pixel in range(B * H * W):
                    output_flat[pixel] = self._apply_kan_single(
                        channel_input_flat[pixel], 
                        weights_flat[pixel]
                    )
                
                # Reshape back to spatial format
                output[:, ch, :, :] = output_flat.view(B, H, W)
        
        return output


class HybridSpectralMixer(nn.Module):
    """
    Hybrid spectral mixer for combining local window processing with global mixing.
    
    Applies lightweight global mixing after local window processing.
    """
    
    def __init__(
        self,
        num_channels: int = 31,
        mixer_type: str = "conv1x1",
        reduction_ratio: int = 4
    ):
        """
        Args:
            num_channels: Number of spectral channels
            mixer_type: Type of global mixer ("conv1x1", "attention")
            reduction_ratio: Channel reduction ratio for attention
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.mixer_type = mixer_type
        
        if mixer_type == "conv1x1":
            self.mixer = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * 2, 1),
                nn.GELU(),
                nn.Conv2d(num_channels * 2, num_channels, 1),
                nn.Sigmoid()
            )
        elif mixer_type == "attention":
            hidden_dim = max(1, num_channels // reduction_ratio)
            self.mixer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(num_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_channels),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"mixer_type must be 'conv1x1' or 'attention', got {mixer_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hybrid spectral mixing.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Mixed tensor of shape [B, C, H, W]
        """
        if self.mixer_type == "conv1x1":
            mixing_weights = self.mixer(x)
            return x * mixing_weights
        else:  # attention
            B, C, H, W = x.shape
            mixing_weights = self.mixer(x).view(B, C, 1, 1)
            return x * mixing_weights
