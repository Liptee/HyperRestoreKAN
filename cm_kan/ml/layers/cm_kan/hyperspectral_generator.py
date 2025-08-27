import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


class MemoryEfficientHyperspectralAttention(nn.Module):
    """
    Memory-efficient attention for hyperspectral images.
    Uses channel attention to reduce memory complexity.
    """
    
    def __init__(self, dim, num_heads, bias, window_size=8):
        super(MemoryEfficientHyperspectralAttention, self).__init__()
        
        self.num_heads = min(num_heads, 2)  # Limit number of heads for memory
        self.dim = dim
        
        # Ensure dim is divisible by num_heads
        if dim % self.num_heads != 0:
            for i in range(self.num_heads, 0, -1):
                if dim % i == 0:
                    self.num_heads = i
                    break
        
        self.head_dim = dim // self.num_heads
        
        # Use depthwise separable convolutions for efficiency
        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        )
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # Simplified illumination modulation
        self.illu_modulation = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape
        
        # Downsample if too large to fit in memory
        original_size = (h, w)
        if h * w > 8192:  # 64x64 or larger
            scale_factor = (8192 / (h * w)) ** 0.5
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            if illu_feat.shape[2:] != x.shape[2:]:
                illu_feat = F.interpolate(illu_feat, size=(new_h, new_w), mode='bilinear', align_corners=False)
            h, w = new_h, new_w
        else:
            if illu_feat.shape[2:] != x.shape[2:]:
                illu_feat = F.interpolate(illu_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # Apply illumination modulation - ensure this is always used
        illu_modulated = self.illu_modulation(illu_feat)
        x_modulated = x + 0.1 * illu_modulated  # Reduce influence
        
        # Simple channel attention instead of spatial attention for memory efficiency
        # Global average pooling
        x_gap = F.adaptive_avg_pool2d(x_modulated, 1)  # (b, c, 1, 1)
        
        # Generate Q, K, V from pooled features
        qkv = self.qkv(x_gap)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Channel attention
        attn = torch.sigmoid(q * k) * self.scale
        out_gap = attn * v
        
        # Broadcast back to spatial dimensions
        out = out_gap.expand_as(x_modulated) * x_modulated + x_modulated
        
        # Upsample back if we downsampled
        if (h, w) != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)

        out = self.project_out(out)
        return out


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class HyperspectralIlluminationEstimator(nn.Module):
    """
    Illumination estimator adapted for hyperspectral images with 31 channels.
    Uses channel-wise attention to handle the high dimensionality.
    """
    def __init__(self, n_fea_middle=64, n_fea_in=31, n_fea_out=31):
        super(HyperspectralIlluminationEstimator, self).__init__()
        
        # First stage: channel reduction with attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_fea_in, n_fea_in // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_fea_in // 4, n_fea_in, 1),
            nn.Sigmoid()
        )
        
        # Spectral mean calculation (instead of simple mean for RGB)
        self.spectral_conv = nn.Conv2d(n_fea_in, 1, kernel_size=1, bias=True)
        
        self.conv1 = nn.Conv2d(n_fea_in + 1, n_fea_middle, kernel_size=1, bias=True)
        
        # Depthwise convolution adapted for hyperspectral
        # Ensure groups divides n_fea_middle evenly
        groups = min(n_fea_middle // 4, n_fea_middle) if n_fea_middle >= 4 else 1
        groups = max(1, groups)
        while n_fea_middle % groups != 0 and groups > 1:
            groups -= 1
            
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=groups,
        )
        
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        
        # Spectral normalization layer
        self.spectral_norm = nn.LayerNorm([n_fea_out])

    def forward(self, img):
        # img: b,c=31,h,w
        # Apply channel attention to emphasize important spectral bands
        channel_att = self.channel_attention(img)
        img_attended = img * channel_att
        
        # Spectral mean calculation (weighted combination of all bands)
        spectral_mean = self.spectral_conv(img_attended)  # b,1,h,w
        
        input_feat = torch.cat([img, spectral_mean], dim=1)  # b,32,h,w
        
        x_1 = self.conv1(input_feat)  # b,64,h,w
        illu_fea = self.depth_conv(x_1)  # b,64,h,w
        illu_map = self.conv2(illu_fea)  # b,31,h,w
        
        # Apply spectral normalization
        b, c, h, w = illu_map.shape
        illu_map = illu_map.permute(0, 2, 3, 1).reshape(-1, c)
        illu_map = self.spectral_norm(illu_map)
        illu_map = illu_map.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        return illu_fea, illu_map


class HyperspectralLayerNorm(nn.Module):
    """Layer normalization adapted for hyperspectral data"""
    def __init__(self, dim):
        super(HyperspectralLayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class HyperspectralAttention(nn.Module):
    """
    Multi-head attention adapted for hyperspectral images.
    Includes spectral-spatial attention mechanism.
    """
    def __init__(self, dim, num_heads, bias):
        super(HyperspectralAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        # Temperature parameters for spectral and spatial attention
        self.temperature_spectral = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_spatial = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Spectral attention branch
        self.q_spectral = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_spectral = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_spectral = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # Spatial attention branch
        self.q_spatial = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, stride=2, 
            padding_mode="reflect", groups=dim, bias=bias
        )
        self.k_spatial = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, stride=2,
            padding_mode="reflect", bias=bias
        )
        self.v_spatial = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # Anchor for combining spectral and spatial information
        self.anchor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2, 
                     padding_mode="reflect", groups=dim, bias=bias),
            nn.Conv2d(dim, dim // 2, kernel_size=1),
        )
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape

        # Spectral attention
        q_spec = self.q_spectral(x)
        k_spec = self.k_spectral(x) 
        v_spec = self.v_spectral(x) * illu_feat
        
        # Spatial attention  
        q_spat = self.q_spatial(x)
        k_spat = self.k_spatial(x)
        v_spat = self.v_spatial(x) * illu_feat
        
        anchor = self.anchor(x)

        # Ensure channels are divisible by num_heads
        def safe_rearrange(tensor, pattern, num_heads):
            b, c, h, w = tensor.shape
            # Make sure c is divisible by num_heads
            if c % num_heads != 0:
                # Pad channels to make it divisible
                pad_channels = num_heads - (c % num_heads)
                tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_channels))
                c = tensor.shape[1]
            return rearrange(tensor, pattern, head=num_heads)

        # Reshape for multi-head attention with safe division
        q_spec = safe_rearrange(q_spec, "b (head c) h w -> b head c (h w)", self.num_heads)
        k_spec = safe_rearrange(k_spec, "b (head c) h w -> b head c (h w)", self.num_heads)
        v_spec = safe_rearrange(v_spec, "b (head c) h w -> b head c (h w)", self.num_heads)
        
        q_spat = safe_rearrange(q_spat, "b (head c) h w -> b head c (h w)", self.num_heads)
        k_spat = safe_rearrange(k_spat, "b (head c) h w -> b head c (h w)", self.num_heads)
        v_spat = safe_rearrange(v_spat, "b (head c) h w -> b head c (h w)", self.num_heads)
        
        anchor = safe_rearrange(anchor, "b (head c) h w -> b head c (h w)", self.num_heads)

        # Normalize
        q_spec = F.normalize(q_spec, dim=-1)
        k_spec = F.normalize(k_spec, dim=-1)
        q_spat = F.normalize(q_spat, dim=-1)
        k_spat = F.normalize(k_spat, dim=-1)
        anchor = F.normalize(anchor, dim=-1)

        # Ensure spatial dimensions match for matrix multiplication
        # Get the spatial dimensions for each tensor
        q_spatial_dim = q_spec.shape[-1]  # H/s * W/s for q
        anchor_spatial_dim = anchor.shape[-1]  # H/s * W/s for anchor  
        v_spec_spatial_dim = v_spec.shape[-1]  # H * W for v_spec
        k_spat_spatial_dim = k_spat.shape[-1]  # H/s * W/s for k_spat
        
        # Spectral attention - ensure dimensions match
        if q_spatial_dim != anchor_spatial_dim:
            min_dim = min(q_spatial_dim, anchor_spatial_dim)
            q_spec = q_spec[..., :min_dim]
            anchor_trunc = anchor[..., :min_dim]
        else:
            anchor_trunc = anchor
            
        attn_spec = (q_spec @ anchor_trunc.transpose(-2, -1)) * self.temperature_spectral
        attn_spec = attn_spec.softmax(dim=-1)
        
        # For v_spec, we need to handle the dimension mismatch
        if v_spec_spatial_dim != q_spatial_dim:
            # Downsample v_spec to match q_spec spatial dimensions
            v_spec_h = int(v_spec_spatial_dim**0.5)
            q_spec_h = int(q_spatial_dim**0.5)
            v_spec_reshaped = v_spec.view(v_spec.shape[0], v_spec.shape[1], v_spec.shape[2], v_spec_h, v_spec_h)
            v_spec_down = F.adaptive_avg_pool2d(v_spec_reshaped, (q_spec_h, q_spec_h))
            v_spec_matched = v_spec_down.view(v_spec.shape[0], v_spec.shape[1], v_spec.shape[2], -1)
        else:
            v_spec_matched = v_spec
            
        # Spatial attention - ensure dimensions match
        if anchor_spatial_dim != k_spat_spatial_dim:
            min_dim = min(anchor_spatial_dim, k_spat_spatial_dim)
            anchor_spat = anchor[..., :min_dim]
            k_spat_matched = k_spat[..., :min_dim]
        else:
            anchor_spat = anchor
            k_spat_matched = k_spat
            
        attn_spat = (anchor_spat @ k_spat_matched.transpose(-2, -1)) * self.temperature_spatial
        attn_spat = attn_spat.softmax(dim=-1)

        # Apply attention with matched dimensions
        out_spec = attn_spec @ v_spec_matched
        out_spat = attn_spat @ v_spat
        
        # Weighted combination - ensure dimensions match
        if out_spec.shape != out_spat.shape:
            # Make them match by using the smaller spatial dimension
            min_spatial_dim = min(out_spec.shape[-1], out_spat.shape[-1])
            out_spec = out_spec[..., :min_spatial_dim]
            out_spat = out_spat[..., :min_spatial_dim]
            
        out = 0.6 * out_spec + 0.4 * out_spat

        # Calculate the actual spatial dimensions from the output
        actual_spatial_dim = out.shape[-1]
        actual_h = int(actual_spatial_dim**0.5)
        actual_w = actual_h
        
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=actual_h, w=actual_w
        )
        
        # If we need to upsample back to original size
        if actual_h != h or actual_w != w:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        # Crop back to original channel size if we padded
        if out.shape[1] > self.dim:
            out = out[:, :self.dim, :, :]

        out = self.project_out(out)
        return out


class HyperspectralFFN(nn.Module):
    """
    Feed-forward Network adapted for hyperspectral processing
    with spectral-aware convolutions
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 2  # Increased for hyperspectral
        
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        
        # Spectral-aware depthwise convolution
        self.depthwise = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, 
            padding=1, dilation=1, groups=hidden_features
        )
        
        # Additional spectral processing layer
        # Ensure groups divides hidden_features evenly
        groups = min(hidden_features // 4, hidden_features) if hidden_features >= 4 else 1
        groups = max(1, groups)  # Ensure at least 1 group
        # Make sure groups divides hidden_features
        while hidden_features % groups != 0 and groups > 1:
            groups -= 1
        
        self.spectral_conv = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=1, groups=groups
        )
        
        self.pointwise2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = nn.GELU()  # GELU works better for hyperspectral data

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.spectral_conv(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class HyperspectralTransformerBlock(nn.Module):
    """
    Transformer block adapted for hyperspectral image processing
    """
    def __init__(self, in_channel, mid_channel, out_channel, num_heads, bias):
        super(HyperspectralTransformerBlock, self).__init__()

        self.norm1 = HyperspectralLayerNorm(in_channel)
        self.attn = MemoryEfficientHyperspectralAttention(in_channel, num_heads, bias)
        self.norm2 = HyperspectralLayerNorm(in_channel)
        self.ffn = HyperspectralFFN(in_channel, mid_channel, out_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))
        return x


class HyperspectralEncoder2D(torch.nn.Module):
    """
    Encoder adapted for hyperspectral images with 31 channels
    Uses hierarchical feature extraction with spectral-spatial attention
    """
    def __init__(self, in_dim=31, out_dim=None, kernel_size=3):
        super(HyperspectralEncoder2D, self).__init__()
        
        if out_dim is None:
            out_dim = in_dim * 21  # Adaptive output dimension
            
        self.estimator = HyperspectralIlluminationEstimator(64, in_dim, in_dim)

        # Multi-scale processing for hyperspectral data
        self.down1 = DWTForward()  # First downsampling: 31*4 = 124 channels
        self.trans1 = HyperspectralTransformerBlock(124, 124, 124, 4, True)
        self.illu_down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(64, 124, 1),
        )

        self.down2 = DWTForward()  # Second downsampling: 124*4 = 496 channels  
        self.trans2 = HyperspectralTransformerBlock(496, 496, 496, 8, True)
        self.illu_down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(64, 496, 1),
        )

        # Additional level for better hyperspectral feature extraction
        self.down3 = DWTForward()  # Third downsampling: 496*4 = 1984 channels
        self.trans3 = HyperspectralTransformerBlock(1984, 1984, 1984, 16, True)
        self.illu_down3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(64, 1984, 1),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up3 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

        # Output projection with spectral attention
        total_channels = in_dim + 124 + 496 + 1984  # 31 + 124 + 496 + 1984 = 2635
        self.conv_out = nn.Sequential(
            HyperspectralLayerNorm(total_channels), 
            HyperspectralFFN(total_channels, total_channels // 2, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: b,31,h,w
        illu_fea, illu_map = self.estimator(x)
        x = x * illu_map + x  # Enhanced input with illumination

        # Multi-scale feature extraction
        x1 = self.down1(x)  # b,124,h/2,w/2
        illu_fea_1 = self.illu_down1(illu_fea)
        x1 = self.trans1(x1, illu_fea_1)

        x2 = self.down2(x1)  # b,496,h/4,w/4
        illu_fea_2 = self.illu_down2(illu_fea)
        x2 = self.trans2(x2, illu_fea_2)

        x3 = self.down3(x2)  # b,1984,h/8,w/8
        illu_fea_3 = self.illu_down3(illu_fea)
        x3 = self.trans3(x3, illu_fea_3)

        # Upsampling and concatenation
        x1 = self.up1(x1)  # b,124,h,w
        x2 = self.up2(x2)  # b,496,h,w  
        x3 = self.up3(x3)  # b,1984,h,w
        
        x = torch.cat([x, x1, x2, x3], dim=1)  # b,2635,h,w
        x = self.conv_out(x)
        return x


class HyperspectralGeneratorLayer(torch.nn.Module):
    """
    Generator layer adapted for hyperspectral images with 31 channels
    Uses spectral basis decomposition for efficient parameter generation
    """
    def __init__(self, in_channels=31, out_channels=None):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels * 10   # Ultra-minimal from 50 for extreme memory efficiency

        # Increased mid channels for hyperspectral complexity
        MID_CHANNELS = in_channels * 2  # 31 * 2 = 62 (ultra-minimal)
        self.encoder = HyperspectralEncoder2D(in_channels, MID_CHANNELS, 3)

        self.norm1 = HyperspectralLayerNorm(MID_CHANNELS)

        # Larger basis for hyperspectral data
        N = 512  # Increased basis size for 31-channel complexity
        self.basis = nn.Parameter(torch.randn(1, MID_CHANNELS, N) * 0.02)

        # Spectral-aware projections
        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)

        # Additional spectral processing
        self.spectral_gate = nn.Sequential(
            nn.Conv2d(MID_CHANNELS, MID_CHANNELS // 4, 1),
            nn.GELU(),
            nn.Conv2d(MID_CHANNELS // 4, MID_CHANNELS, 1),
            nn.Sigmoid()
        )

        self.norm2 = HyperspectralLayerNorm(MID_CHANNELS)
        self.conv_reproj = HyperspectralFFN(
            in_features=MID_CHANNELS, 
            hidden_features=MID_CHANNELS // 2,
            out_features=out_channels
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Forward projection through hyperspectral encoder
        x = self.encoder(x)  # b,2635,h,w
        x = self.norm1(x)

        # Spectral gating for enhanced feature selection
        gate = self.spectral_gate(x)
        x = x * gate

        # Basis coefficient computation
        q = self.q(x)  # b,2635,h,w
        k = self.k(self.basis)  # b,2635,512
        v = self.v(self.basis)  # b,2635,512

        q = rearrange(q, "b c h w -> b c (h w)")

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention mechanism for basis selection
        a = (q.transpose(-2, -1) @ k).transpose(-2, -1)  # b,2635,hw
        a = F.relu(a)
        
        # Apply spectral weighting
        spectral_weights = torch.softmax(a.mean(dim=-1, keepdim=True), dim=1)
        a = a * spectral_weights

        y = v @ a  # b,2635,hw
        y = rearrange(y, "b c (h w) -> b c h w", h=H, w=W)

        # Back projection
        x = self.norm2(y)
        x = self.conv_reproj(x)

        return x


class LightHyperspectralEncoder2D(torch.nn.Module):
    """
    Lightweight version of hyperspectral encoder for faster inference
    """
    def __init__(self, in_dim=31, out_dim=None, kernel_size=3):
        super(LightHyperspectralEncoder2D, self).__init__()
        
        if out_dim is None:
            out_dim = in_dim * 5  # Reduced for lightweight version
            
        self.estimator = HyperspectralIlluminationEstimator(32, in_dim, in_dim)

        # Single-scale processing for efficiency
        self.down1 = DWTForward()  # 31*4 = 124 channels
        self.trans1 = HyperspectralTransformerBlock(124, 124, 124, 2, True)
        self.illu_down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(32, 124, 1),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Reduced output channels
        total_channels = in_dim + 124  # 31 + 124 = 155
        self.conv_out = nn.Sequential(
            HyperspectralLayerNorm(total_channels), 
            HyperspectralFFN(total_channels, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(x)
        x = x * illu_map + x

        x1 = self.down1(x)
        illu_fea = self.illu_down1(illu_fea)
        x1 = self.trans1(x1, illu_fea)

        x1 = self.up1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_out(x)
        return x


class LightHyperspectralGeneratorLayer(torch.nn.Module):
    """
    Lightweight hyperspectral generator for faster training/inference
    """
    def __init__(self, in_channels=31, out_channels=None):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels * 50  # Reduced parameter count

        MID_CHANNELS = in_channels * 5  # 31 * 5 = 155
        self.encoder = LightHyperspectralEncoder2D(in_channels, MID_CHANNELS, 3)

        self.norm1 = HyperspectralLayerNorm(MID_CHANNELS)

        # Smaller basis for efficiency
        N = 128
        self.basis = nn.Parameter(torch.randn(1, MID_CHANNELS, N) * 0.02)

        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)

        self.norm2 = HyperspectralLayerNorm(MID_CHANNELS)
        self.conv_reproj = HyperspectralFFN(
            in_features=MID_CHANNELS, 
            out_features=out_channels
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        x = self.encoder(x)
        x = self.norm1(x)

        q = self.q(x)
        k = self.k(self.basis)
        v = self.v(self.basis)

        q = rearrange(q, "b c h w -> b c (h w)")

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        a = (q.transpose(-2, -1) @ k).transpose(-2, -1)
        a = F.relu(a)

        y = v @ a
        y = rearrange(y, "b c (h w) -> b c h w", h=H, w=W)

        x = self.norm2(y)
        x = self.conv_reproj(x)

        return x
