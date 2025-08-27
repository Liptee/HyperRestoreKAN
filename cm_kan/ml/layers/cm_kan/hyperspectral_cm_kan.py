import numpy as np
import torch
from .kan import KANLayer
from .hyperspectral_generator import HyperspectralGeneratorLayer, LightHyperspectralGeneratorLayer


class HyperspectralCmKANLayer(torch.nn.Module):
    """
    Hyperspectral Color Matching KAN Layer for 31-channel processing
    Adapted from the original CmKANLayer to handle hyperspectral data
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
        super(HyperspectralCmKANLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # KAN layer for hyperspectral processing
        self.kan_layer = KANLayer(
            in_dim=in_channels,
            out_dim=out_channels,
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
        self.generator = HyperspectralGeneratorLayer(in_channels, self.kan_params_num)

    def kan(self, x, w):
        """Apply KAN transformation with generated weights"""
        i, j = self.kan_params_indices[0], self.kan_params_indices[1]
        coef = w[:, i:j].view(-1, *self.kan_layer.activation_fn.coef_shape)
        
        i, j = self.kan_params_indices[1], self.kan_params_indices[2]
        univariate_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.univariate_weight_shape
        )
        
        i, j = self.kan_params_indices[2], self.kan_params_indices[3]
        residual_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.residual_weight_shape
        )
        
        x = self.kan_layer(x, coef, univariate_weight, residual_weight)
        return x.squeeze(0)

    def forward(self, x):
        """
        Forward pass for hyperspectral data
        Input: x with shape (B, 31, H, W)
        Output: processed hyperspectral image with shape (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # Validate input channels
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

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
