import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperspectralPatchDiscriminator(nn.Module):
    """
    Patch discriminator adapted for hyperspectral images with 31 channels
    Uses spectral-aware convolutions and attention mechanisms
    """
    
    def __init__(self, in_dim=31, ndf=64, n_layers=3, use_spectral_attention=True):
        super(HyperspectralPatchDiscriminator, self).__init__()
        
        self.use_spectral_attention = use_spectral_attention
        
        # Initial spectral processing
        if use_spectral_attention:
            self.spectral_attention = SpectralAttentionBlock(in_dim)
        
        # Build discriminator layers
        layers = []
        
        # First layer: 31 -> ndf
        layers.append(nn.Conv2d(in_dim, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            layers.append(nn.Conv2d(
                ndf * nf_mult_prev, 
                ndf * nf_mult,
                kernel_size=4, 
                stride=2, 
                padding=1
            ))
            layers.append(nn.InstanceNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        layers.append(nn.Conv2d(
            ndf * nf_mult_prev, 
            ndf * nf_mult,
            kernel_size=4, 
            stride=1, 
            padding=1
        ))
        layers.append(nn.InstanceNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for hyperspectral patch discrimination
        
        Args:
            x (torch.Tensor): Input hyperspectral image (B, 31, H, W)
            
        Returns:
            torch.Tensor: Patch-wise discrimination scores
        """
        if self.use_spectral_attention:
            x = self.spectral_attention(x)
        
        return self.model(x)


class SpectralAttentionBlock(nn.Module):
    """
    Spectral attention mechanism for hyperspectral discriminator
    Helps focus on discriminative spectral bands
    """
    
    def __init__(self, channels=31, reduction=4):
        super(SpectralAttentionBlock, self).__init__()
        
        self.channels = channels
        
        # Global spectral pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Spectral attention network
        self.attention_net = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spectral correlation analysis
        self.correlation_conv = nn.Conv2d(channels, channels, 1, groups=channels)

    def forward(self, x):
        """
        Apply spectral attention to input
        """
        B, C, H, W = x.shape
        
        # Global spectral attention
        global_feat = self.global_pool(x)  # (B, C, 1, 1)
        attention_weights = self.attention_net(global_feat)  # (B, C, 1, 1)
        
        # Apply attention
        x_attended = x * attention_weights
        
        # Spectral correlation enhancement
        x_corr = self.correlation_conv(x_attended)
        
        # Combine original and enhanced features
        output = x + 0.1 * x_corr  # Small residual connection
        
        return output


class MultiScaleHyperspectralDiscriminator(nn.Module):
    """
    Multi-scale discriminator for hyperspectral images
    Operates at multiple spatial resolutions for better discrimination
    """
    
    def __init__(self, in_dim=31, ndf=64, num_scales=3):
        super(MultiScaleHyperspectralDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        
        # Create discriminators for different scales
        self.discriminators = nn.ModuleList([
            HyperspectralPatchDiscriminator(
                in_dim=in_dim,
                ndf=ndf // (2**i),  # Reduce capacity for smaller scales
                n_layers=3,
                use_spectral_attention=True
            ) for i in range(num_scales)
        ])
        
        # Downsampling layers
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Multi-scale discrimination
        
        Returns:
            list: Discrimination results at different scales
        """
        results = []
        current_x = x
        
        for i, discriminator in enumerate(self.discriminators):
            result = discriminator(current_x)
            results.append(result)
            
            # Downsample for next scale (except for the last one)
            if i < self.num_scales - 1:
                current_x = self.downsample(current_x)
        
        return results


class SpectralGAN_Discriminator(nn.Module):
    """
    Advanced discriminator with spectral normalization and self-attention
    Designed specifically for high-quality hyperspectral image generation
    """
    
    def __init__(self, in_dim=31, ndf=64, use_self_attention=True):
        super(SpectralGAN_Discriminator, self).__init__()
        
        self.use_self_attention = use_self_attention
        
        # Spectral normalization for stable training
        def spectral_norm(layer):
            return nn.utils.spectral_norm(layer)
        
        # Initial convolution
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(in_dim, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downsampling blocks
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self-attention block
        if use_self_attention:
            self.self_attention = SelfAttentionBlock(ndf * 4)
        
        self.down3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output layer
        self.output = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, x):
        """
        Forward pass with spectral normalization and self-attention
        """
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        if self.use_self_attention:
            x = self.self_attention(x)
        
        x = self.down3(x)
        x = self.output(x)
        
        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-attention mechanism for discriminator
    Helps capture long-range dependencies in hyperspectral images
    """
    
    def __init__(self, channels):
        super(SelfAttentionBlock, self).__init__()
        
        self.channels = channels
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.out = nn.Conv2d(channels, channels, 1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Apply self-attention
        """
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        k = self.key(x).view(B, -1, H * W)  # (B, C//8, HW)
        v = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        
        # Compute attention
        attention = torch.bmm(q, k)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        
        # Output projection and residual connection
        out = self.out(out)
        out = self.gamma * out + x
        
        return out


class HyperspectralFeatureMatchingDiscriminator(nn.Module):
    """
    Discriminator with feature matching capability
    Returns intermediate features for feature matching loss
    """
    
    def __init__(self, in_dim=31, ndf=64):
        super(HyperspectralFeatureMatchingDiscriminator, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(ndf * 8, 1, 4, 1, 1)

    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature return
        
        Args:
            x: Input hyperspectral image
            return_features: If True, returns intermediate features
            
        Returns:
            If return_features=False: discrimination result
            If return_features=True: (discrimination result, list of features)
        """
        features = []
        
        x = self.conv1(x)
        if return_features:
            features.append(x)
        
        x = self.conv2(x)
        if return_features:
            features.append(x)
        
        x = self.conv3(x)
        if return_features:
            features.append(x)
        
        x = self.conv4(x)
        if return_features:
            features.append(x)
        
        output = self.classifier(x)
        
        if return_features:
            return output, features
        else:
            return output
