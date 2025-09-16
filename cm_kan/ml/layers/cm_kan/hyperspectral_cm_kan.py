import numpy as np
import torch
from .kan import KANLayer
from .hyperspectral_generator import HyperspectralGeneratorLayer, LightHyperspectralGeneratorLayer
from .spectral_window import SpectralWindowPreprocessor, GroupedKANLayer, HybridSpectralMixer
from typing import Optional, List


class HyperspectralCmKANLayer(torch.nn.Module):
    """
    Hyperspectral Color Matching KAN Layer for 31-channel processing
    Adapted from the original CmKANLayer to handle hyperspectral data
    
    Supports multiple spectral processing modes:
    - global: Traditional global spectral processing (default)
    - local_window: Local spectral window processing
    - hybrid: Local window + global mixing
    """
    def __init__(
        self,
        in_channels=31,
        out_channels=31,
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=(-1.0, 1.0),
        spectral_mode="global",
        window_size=3,
        padding_mode="reflect",
        wavelengths=None,
        shared_kan_params=True,
    ):
        super(HyperspectralCmKANLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_mode = spectral_mode
        self.window_size = window_size
        self.padding_mode = padding_mode
        self.wavelengths = wavelengths
        self.shared_kan_params = shared_kan_params
        
        # Validate spectral mode
        if spectral_mode not in ["global", "local_window", "hybrid"]:
            raise ValueError(f"spectral_mode must be one of ['global', 'local_window', 'hybrid'], got {spectral_mode}")
        
        if spectral_mode == "global":
            # Traditional global processing
            self._setup_global_mode(grid_size, spline_order, residual_std, grid_range)
        elif spectral_mode == "local_window":
            # Local window processing
            self._setup_local_window_mode(grid_size, spline_order, residual_std, grid_range)
        else:  # hybrid
            # Local window + global mixing
            self._setup_hybrid_mode(grid_size, spline_order, residual_std, grid_range)

    def _setup_global_mode(self, grid_size, spline_order, residual_std, grid_range):
        """Setup traditional global spectral processing mode."""
        # KAN layer for hyperspectral processing
        self.kan_layer = KANLayer(
            in_dim=self.in_channels,
            out_dim=self.out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )

        # Calculate KAN parameters for hyperspectral data
        self.kan_params_num = 0
        self.kan_params_indices = [0]

        coef_len = np.prod(self.kan_layer.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.kan_layer.residual_layer.univariate_weight_shape
        )
        residual_weight_len = np.prod(
            self.kan_layer.residual_layer.residual_weight_shape
        )
        
        self.kan_params_indices.extend(
            [coef_len, univariate_weight_len, residual_weight_len]
        )

        self.kan_params_num = np.sum(self.kan_params_indices)
        self.kan_params_indices = np.cumsum(self.kan_params_indices)

        # Hyperspectral generator for parameter generation
        self.generator = HyperspectralGeneratorLayer(self.in_channels, self.kan_params_num)

    def _setup_local_window_mode(self, grid_size, spline_order, residual_std, grid_range):
        """Setup local spectral window processing mode."""
        # Window preprocessor
        self.window_preprocessor = SpectralWindowPreprocessor(
            num_channels=self.in_channels,
            window_size=self.window_size,
            padding_mode=self.padding_mode,
            wavelengths=self.wavelengths
        )
        
        # Grouped KAN for window processing
        self.grouped_kan = GroupedKANLayer(
            num_channels=self.out_channels,
            window_size=self.window_size,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
            shared_params=self.shared_kan_params
        )

    def _setup_hybrid_mode(self, grid_size, spline_order, residual_std, grid_range):
        """Setup hybrid mode with local window + global mixing."""
        # Setup local window processing
        self._setup_local_window_mode(grid_size, spline_order, residual_std, grid_range)
        
        # Add global spectral mixer
        self.global_mixer = HybridSpectralMixer(
            num_channels=self.out_channels,
            mixer_type="conv1x1"
        )

    def kan(self, x, w):
        """Apply KAN transformation with generated weights"""
        # Process each sample in the batch individually
        batch_size = x.shape[0]
        outputs = []
        
        for b in range(batch_size):
            # Extract parameters for this sample
            i, j = self.kan_params_indices[0], self.kan_params_indices[1]
            coef = w[b, i:j].view(*self.kan_layer.activation_fn.coef_shape)
            
            i, j = self.kan_params_indices[1], self.kan_params_indices[2]
            univariate_weight = w[b, i:j].view(
                *self.kan_layer.residual_layer.univariate_weight_shape
            )
            
            i, j = self.kan_params_indices[2], self.kan_params_indices[3]
            residual_weight = w[b, i:j].view(
                *self.kan_layer.residual_layer.residual_weight_shape
            )
            
            # Apply KAN to this sample
            sample_output = self.kan_layer(x[b:b+1], coef, univariate_weight, residual_weight)
            outputs.append(sample_output)
        
        # Concatenate outputs
        return torch.cat(outputs, dim=0)

    def forward(self, x):
        """
        Forward pass for hyperspectral data with support for different spectral modes.
        
        Args:
            x: Input tensor with shape (B, in_channels, H, W)
            
        Returns:
            Processed hyperspectral image with shape (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # Validate input channels
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        if self.spectral_mode == "global":
            return self._forward_global(x)
        elif self.spectral_mode == "local_window":
            return self._forward_local_window(x)
        else:  # hybrid
            return self._forward_hybrid(x)

    def _forward_global(self, x):
        """Forward pass for global spectral processing mode."""
        B, C, H, W = x.shape
        
        # Generate KAN weights using hyperspectral generator
        # weights shape: (B, kan_params_num, H, W)
        weights = self.generator(x)
        
        # Reshape weights for pixel-wise processing
        # weights shape: (B * H * W, kan_params_num)
        weights = weights.permute(0, 2, 3, 1)
        weights = weights.reshape(B * H * W, self.kan_params_num)

        # Reshape input for pixel-wise processing
        # x shape: (B * H * W, in_channels)
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Apply KAN transformation pixel-wise
        # x shape: (B * H * W, out_channels)
        x = self.kan(x, weights)

        # Reshape back to image format
        # x shape: (B, out_channels, H, W)
        x = x.view(B, H, W, self.kan_layer.out_dim).permute(0, 3, 1, 2)

        return x

    def _forward_local_window(self, x):
        """Forward pass for local window spectral processing mode."""
        # Apply spectral windowing: [B, C, H, W] -> [B, C, (2n+1), H, W]
        windowed_x = self.window_preprocessor(x)
        
        # Apply grouped KAN processing: [B, C, (2n+1), H, W] -> [B, C, H, W]
        output = self.grouped_kan(windowed_x)
        
        return output

    def _forward_hybrid(self, x):
        """Forward pass for hybrid spectral processing mode."""
        # First apply local window processing
        output = self._forward_local_window(x)
        
        # Then apply global spectral mixing
        output = self.global_mixer(output)
        
        return output


class LightHyperspectralCmKANLayer(HyperspectralCmKANLayer):
    """
    Lightweight version of HyperspectralCmKANLayer for faster processing
    Uses the lightweight generator for reduced computational cost
    """
    def __init__(
        self,
        in_channels=31,
        out_channels=31,
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=(-1.0, 1.0),
    ):
        super(LightHyperspectralCmKANLayer, self).__init__(
            in_channels, out_channels, grid_size, spline_order, residual_std, grid_range
        )
        
        # Replace with lightweight generator
        self.generator = LightHyperspectralGeneratorLayer(in_channels, self.kan_params_num)


class SpectralBandProcessor(torch.nn.Module):
    """
    Specialized processor for handling individual spectral bands
    Can be used as a component in larger hyperspectral architectures
    """
    def __init__(self, num_bands=31, processing_dim=64):
        super(SpectralBandProcessor, self).__init__()
        self.num_bands = num_bands
        self.processing_dim = processing_dim
        
        # Individual band processors
        self.band_processors = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(1, processing_dim, 3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(processing_dim, processing_dim, 3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(processing_dim, 1, 1),
            ) for _ in range(num_bands)
        ])
        
        # Cross-band interaction
        self.cross_band_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(num_bands, num_bands * 2, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(num_bands * 2, num_bands, 1),
        )

    def forward(self, x):
        """
        Process each spectral band individually then fuse
        Input: (B, 31, H, W)
        Output: (B, 31, H, W)
        """
        B, C, H, W = x.shape
        
        # Process each band individually
        processed_bands = []
        for i in range(self.num_bands):
            band = x[:, i:i+1, :, :]  # (B, 1, H, W)
            processed_band = self.band_processors[i](band)
            processed_bands.append(processed_band)
        
        # Combine processed bands
        processed = torch.cat(processed_bands, dim=1)  # (B, 31, H, W)
        
        # Cross-band fusion
        fused = self.cross_band_fusion(processed)
        
        # Residual connection
        output = x + fused
        
        return output


class HyperspectralResidualBlock(torch.nn.Module):
    """
    Residual block specifically designed for hyperspectral processing
    Combines spectral and spatial processing with skip connections
    """
    def __init__(self, channels=31, mid_channels=64):
        super(HyperspectralResidualBlock, self).__init__()
        
        # Calculate safe groups for spectral branch
        spectral_groups = min(mid_channels // 4, mid_channels) if mid_channels >= 4 else 1
        spectral_groups = max(1, spectral_groups)
        while mid_channels % spectral_groups != 0 and spectral_groups > 1:
            spectral_groups -= 1
        
        # Calculate safe groups for spatial branch
        spatial_groups = min(mid_channels // 4, mid_channels) if mid_channels >= 4 else 1
        spatial_groups = max(1, spatial_groups)
        while mid_channels % spatial_groups != 0 and spatial_groups > 1:
            spatial_groups -= 1
        
        # Spectral processing branch
        self.spectral_branch = torch.nn.Sequential(
            torch.nn.Conv2d(channels, mid_channels, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(mid_channels, mid_channels, 1, groups=spectral_groups),
            torch.nn.GELU(),
            torch.nn.Conv2d(mid_channels, channels, 1),
        )
        
        # Spatial processing branch
        self.spatial_branch = torch.nn.Sequential(
            torch.nn.Conv2d(channels, mid_channels, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=spatial_groups),
            torch.nn.GELU(),
            torch.nn.Conv2d(mid_channels, channels, 3, padding=1),
        )
        
        # Feature fusion
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(channels * 2, channels, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        """
        Forward pass with spectral-spatial processing
        """
        spectral_out = self.spectral_branch(x)
        spatial_out = self.spatial_branch(x)
        
        # Combine spectral and spatial features
        combined = torch.cat([spectral_out, spatial_out], dim=1)
        fused = self.fusion(combined)
        
        # Residual connection
        return x + fused


class AdaptiveSpectralNormalization(torch.nn.Module):
    """
    Adaptive normalization for hyperspectral data
    Normalizes across spectral dimension with learned parameters
    """
    def __init__(self, num_channels=31, eps=1e-5):
        super(AdaptiveSpectralNormalization, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        
        # Learnable scale and shift parameters
        self.gamma = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
        # Adaptive weights for each channel
        self.channel_weights = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x):
        """
        Apply adaptive spectral normalization
        """
        # Calculate statistics across spatial dimensions
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable transformation with channel-specific weights
        output = self.gamma * x_norm * self.channel_weights + self.beta
        
        return output
