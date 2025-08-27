import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from cm_kan.ml.layers.cm_kan.hyperspectral_cm_kan import (
    HyperspectralCmKANLayer, 
    LightHyperspectralCmKANLayer,
    SpectralBandProcessor,
    HyperspectralResidualBlock,
    AdaptiveSpectralNormalization
)
from cm_kan.core import Logger


class HyperspectralCmKAN(torch.nn.Module):
    """
    Hyperspectral Color Matching KAN for 31-channel image restoration
    
    This model is specifically designed for hyperspectral image processing with:
    - 31 input/output channels representing different spectral bands
    - Spectral-spatial attention mechanisms
    - Adaptive normalization for hyperspectral data
    - Multi-scale feature processing
    """

    def __init__(
        self, 
        in_dims=[31], 
        out_dims=[31], 
        grid_size=5, 
        spline_order=3, 
        residual_std=0.1, 
        grid_range=(-1.0, 1.0),
        use_spectral_processor=True,
        use_residual_blocks=True,
        num_residual_blocks=2,
        use_gradient_checkpointing=True
    ):
        super(HyperspectralCmKAN, self).__init__()

        Logger.info(f"HyperspectralCmKAN: in_dims={in_dims}, out_dims={out_dims}")
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.use_spectral_processor = use_spectral_processor
        self.use_residual_blocks = use_residual_blocks
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Validate hyperspectral dimensions
        if in_dims[0] != 31:
            Logger.warning(f"Expected 31 input channels for hyperspectral, got {in_dims[0]}")
        if out_dims[0] != 31:
            Logger.warning(f"Expected 31 output channels for hyperspectral, got {out_dims[0]}")

        # Input preprocessing for hyperspectral data
        self.input_norm = AdaptiveSpectralNormalization(in_dims[0])
        
        # Optional spectral band processor
        if use_spectral_processor:
            self.spectral_processor = SpectralBandProcessor(
                num_bands=in_dims[0], 
                processing_dim=64
            )

        # Main KAN layers for hyperspectral processing
        cm_kan_size = [s for s in zip(in_dims, out_dims)]
        self.layers = []
        
        for i, (in_dim, out_dim) in enumerate(cm_kan_size):
            layer = HyperspectralCmKANLayer(
                in_channels=in_dim,
                out_channels=out_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                residual_std=residual_std,
                grid_range=grid_range,
            )
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        # Optional residual blocks for enhanced feature processing
        if use_residual_blocks:
            self.residual_blocks = nn.ModuleList([
                HyperspectralResidualBlock(channels=out_dims[-1], mid_channels=128)
                for _ in range(num_residual_blocks)
            ])

        # Output post-processing
        self.output_norm = AdaptiveSpectralNormalization(out_dims[-1])
        
        # Final refinement layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_dims[-1], out_dims[-1] * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dims[-1] * 2, out_dims[-1], 3, padding=1),
            nn.Sigmoid()  # Ensure output is in valid range
        )

    def forward(self, x):
        """
        Forward pass for hyperspectral image restoration
        
        Args:
            x (torch.Tensor): Input hyperspectral image with shape (B, 31, H, W)
            
        Returns:
            torch.Tensor: Restored hyperspectral image with shape (B, 31, H, W)
        """
        # Store input for residual connection
        input_residual = x
        
        # Input normalization
        x = self.input_norm(x)
        
        # Optional spectral preprocessing
        if self.use_spectral_processor:
            x = self.spectral_processor(x)
        
        # Apply KAN layers with optional gradient checkpointing
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        # Apply residual blocks if enabled
        if self.use_residual_blocks:
            for res_block in self.residual_blocks:
                if self.use_gradient_checkpointing and self.training:
                    x = checkpoint(res_block, x, use_reentrant=False)
                else:
                    x = res_block(x)
        
        # Output normalization
        x = self.output_norm(x)
        
        # Final refinement
        refinement = self.final_conv(x)
        x = x * refinement  # Element-wise modulation
        
        # Global residual connection
        x = x + input_residual
        
        return x


class LightHyperspectralCmKAN(torch.nn.Module):
    """
    Lightweight version of HyperspectralCmKAN for faster processing
    Suitable for real-time applications or resource-constrained environments
    """

    def __init__(
        self, 
        in_dims=[31], 
        out_dims=[31], 
        grid_size=3, 
        spline_order=2, 
        residual_std=0.05, 
        grid_range=(-1.0, 1.0)
    ):
        super(LightHyperspectralCmKAN, self).__init__()

        Logger.info(f"LightHyperspectralCmKAN: in_dims={in_dims}, out_dims={out_dims}")

        self.in_dims = in_dims
        self.out_dims = out_dims

        # Simplified input processing
        self.input_norm = AdaptiveSpectralNormalization(in_dims[0])

        # Lightweight KAN layers
        cm_kan_size = [s for s in zip(in_dims, out_dims)]
        self.layers = []
        
        for in_dim, out_dim in cm_kan_size:
            layer = LightHyperspectralCmKANLayer(
                in_channels=in_dim,
                out_channels=out_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                residual_std=residual_std,
                grid_range=grid_range,
            )
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        # Simplified output processing
        self.output_conv = nn.Conv2d(out_dims[-1], out_dims[-1], 3, padding=1)

    def forward(self, x):
        """
        Lightweight forward pass for hyperspectral processing
        """
        input_residual = x
        
        x = self.input_norm(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_conv(x)
        x = torch.sigmoid(x)  # Ensure valid output range
        
        # Residual connection
        x = x + input_residual
        
        return x


class HyperspectralCycleCmKAN(torch.nn.Module):
    """
    Cycle-consistent hyperspectral KAN for unsupervised domain adaptation
    Useful for hyperspectral image translation between different sensors or conditions
    """
    
    def __init__(
        self, 
        in_dims=[31], 
        out_dims=[31], 
        grid_size=5, 
        spline_order=3, 
        residual_std=0.1, 
        grid_range=(-1.0, 1.0)
    ):
        super(HyperspectralCycleCmKAN, self).__init__()

        Logger.info(f"HyperspectralCycleCmKAN: in_dims={in_dims}, out_dims={out_dims}")

        # Forward generator (A -> B)
        self.gen_ab = HyperspectralCmKAN(
            in_dims=in_dims,
            out_dims=out_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )
        
        # Backward generator (B -> A)
        self.gen_ba = HyperspectralCmKAN(
            in_dims=out_dims,
            out_dims=in_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )

        # Import here to avoid circular imports
        from cm_kan.ml.layers import HyperspectralPatchDiscriminator
        self.dis_a = HyperspectralPatchDiscriminator(in_dim=in_dims[0])
        self.dis_b = HyperspectralPatchDiscriminator(in_dim=out_dims[0])

    def forward(self, x, direction='ab'):
        """
        Forward pass for cycle-consistent translation
        
        Args:
            x: Input hyperspectral image
            direction: 'ab' for A->B translation, 'ba' for B->A translation
        """
        if direction == 'ab':
            return self.gen_ab(x)
        elif direction == 'ba':
            return self.gen_ba(x)
        else:
            raise ValueError("Direction must be 'ab' or 'ba'")

    def cycle_forward(self, x_a, x_b):
        """
        Complete cycle-consistent forward pass
        
        Returns:
            dict: Contains all generated images and cycle-consistent reconstructions
        """
        # A -> B -> A
        fake_b = self.gen_ab(x_a)
        cycle_a = self.gen_ba(fake_b)
        
        # B -> A -> B
        fake_a = self.gen_ba(x_b)
        cycle_b = self.gen_ab(fake_a)
        
        return {
            'fake_a': fake_a,
            'fake_b': fake_b,
            'cycle_a': cycle_a,
            'cycle_b': cycle_b,
        }


class MultiScaleHyperspectralCmKAN(torch.nn.Module):
    """
    Multi-scale hyperspectral KAN for processing images at different resolutions
    Useful for handling hyperspectral images with varying spatial resolutions
    """
    
    def __init__(
        self, 
        in_dims=[31], 
        out_dims=[31], 
        scales=[1.0, 0.5, 0.25],
        grid_size=5, 
        spline_order=3, 
        residual_std=0.1, 
        grid_range=(-1.0, 1.0)
    ):
        super(MultiScaleHyperspectralCmKAN, self).__init__()

        Logger.info(f"MultiScaleHyperspectralCmKAN: scales={scales}")

        self.scales = scales
        
        # Create KAN for each scale
        self.scale_networks = nn.ModuleList([
            HyperspectralCmKAN(
                in_dims=in_dims,
                out_dims=out_dims,
                grid_size=grid_size,
                spline_order=spline_order,
                residual_std=residual_std,
                grid_range=grid_range,
                use_residual_blocks=True,
                num_residual_blocks=1
            ) for _ in scales
        ])
        
        # Feature fusion network
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dims[-1] * len(scales), out_dims[-1] * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dims[-1] * 2, out_dims[-1], 3, padding=1),
        )

    def forward(self, x):
        """
        Multi-scale forward pass
        """
        B, C, H, W = x.shape
        scale_outputs = []
        
        # Process at each scale
        for i, (scale, network) in enumerate(zip(self.scales, self.scale_networks)):
            if scale != 1.0:
                # Downsample
                scaled_h, scaled_w = int(H * scale), int(W * scale)
                x_scaled = F.interpolate(x, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Process at this scale
            output_scaled = network(x_scaled)
            
            # Upsample back to original size
            if scale != 1.0:
                output_scaled = F.interpolate(output_scaled, size=(H, W), mode='bilinear', align_corners=False)
            
            scale_outputs.append(output_scaled)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=1)
        output = self.fusion_conv(fused)
        
        # Residual connection
        output = output + x
        
        return output
