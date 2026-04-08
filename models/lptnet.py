# -*- coding: utf-8 -*-
"""
LPTNet: Modeling Dataset-level Priors with Learnable Probability Tables for Pansharpening

This module implements the LPTNet network architecture as described in the paper.
"""
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OverlapPatchEmbed(nn.Module):
    """Patch Embedding module using overlapping convolution."""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """Downsampling module using pixel unshuffle."""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module using pixel shuffle."""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class LearnableProbabilityTable(nn.Module):
    """
    Learnable Probability Table (LPT) module.

    Implements the dataset-level learnable probability table that captures prior
    knowledge across the dataset for pansharpening.
    """
    def __init__(self, dim1, dim, bias):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        # MS branch projections
        self.ms_kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.ms_kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1,
                                       padding=1, groups=dim * 2, bias=bias)

        # Probability and value projections
        self.p_project = nn.Linear(dim1, dim)
        self.v_project = nn.Linear(dim1, dim)

        # Update scales
        self.scale_v = 0.95
        self.scale_p = 0.85

    def forward(self, x, p, v):
        """
        Args:
            x: Input feature map (B, C, H, W)
            p: Probability table (B, N, d)
            v: Value table (B, N, d)
        Returns:
            out: Output feature map (B, C, H, W)
            p: Updated probability table (B, N, d)
            v: Updated value table (B, N, d)
        """
        b, c, h, w = x.shape

        # Generate MS query and value
        ms_qkv = self.ms_kv(x)
        ms_qkv = self.ms_kv_dwconv(ms_qkv)
        ms_q, ms_v = ms_qkv.chunk(2, dim=1)
        ms_q = rearrange(ms_q, 'b c h w -> b (h w) c', h=h, w=w)
        ms_v = rearrange(ms_v, 'b c h w -> b (h w) c', h=h, w=w)

        # Project probability and value tables
        p = self.p_project(p)
        v = self.v_project(v)

        # Compute attention (probability lookup)
        attn_ori = ms_q @ p.transpose(-2, -1) * self.temperature  # B HW N
        attn = attn_ori.softmax(dim=-1)
        out = attn @ v  # B HW d

        # Update probability and value tables
        attn = attn_ori.transpose(-2, -1).softmax(dim=-1)
        v = self.scale_v * v + (1 - self.scale_v) * attn @ ms_v
        p = self.scale_p * p + (1 - self.scale_p) * attn @ ms_q

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        return out, p, v


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def to_3d(x):
    """Convert 4D tensor (B, C, H, W) to 3D (B, HW, C)."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Convert 3D tensor (B, HW, C) to 4D (B, C, H, W)."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    """Layer normalization for 4D tensors."""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class TokenAdaptiveTransformer(nn.Module):
    """
    Token Adaptive Transformer (TAT) module.

    Integrates the Learnable Probability Table with feed-forward network
    for token-adaptive feature transformation.
    """
    def __init__(self, dim1, dim, LayerNorm_type, ffn_expansion_factor, bias):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = LearnableProbabilityTable(dim1, dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, p, v):
        m, p, v = self.attn(self.norm1(x), p, v)
        x = x + m
        x = x + self.ffn(self.norm2(x))
        return x, p, v


class MultiScaleLearnableProbabilityBlock(nn.Module):
    """
    Multi-Scale Learnable Probability Block (MSLPB).

    The main network architecture that employs Token Adaptive Transformers
    at multiple scales for pansharpening.
    """
    def __init__(self, inp_channels, num_tokens=16, dim=48, num_blocks=None,
                 ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False):
        super().__init__()

        # Learnable probability and value tables
        self.q = nn.Parameter(torch.randn(num_tokens, dim))
        self.v = nn.Parameter(torch.randn(num_tokens, dim))

        # Patch embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels + 1, dim)

        # Encoder
        self.encoder_level1 = TokenAdaptiveTransformer(
            dim1=dim, dim=dim, ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type)
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = TokenAdaptiveTransformer(
            dim1=dim, dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type)
        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = TokenAdaptiveTransformer(
            dim1=int(dim * 2 ** 1), dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type)
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        # Latent
        self.latent = TokenAdaptiveTransformer(
            dim1=int(dim * 2 ** 2), dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type)

        # Decoder
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = TokenAdaptiveTransformer(
            dim1=int(dim * 2 ** 3), dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type)

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = TokenAdaptiveTransformer(
            dim1=int(dim * 2 ** 2), dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type)

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = TokenAdaptiveTransformer(
            dim1=int(dim * 2 ** 1), dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type)

        # Output
        self.output = nn.Conv2d(int(dim * 2 ** 1), inp_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, ms, pan):
        """
        Args:
            ms: Multi-spectral image (B, C, H, W)
            pan: Panchromatic image (B, 1, H, W)
        Returns:
            out: Fused pansharpened image (B, C, H, W)
        """
        inp_img = torch.cat([ms, pan], dim=1)
        B = ms.shape[0]

        # Initialize probability and value tables
        p = self.q.unsqueeze(0).expand(B, -1, -1)
        v = self.v.unsqueeze(0).expand(B, -1, -1)

        # Encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1, p, v = self.encoder_level1(inp_enc_level1, p, v)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, p, v = self.encoder_level2(inp_enc_level2, p, v)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, p, v = self.encoder_level3(inp_enc_level3, p, v)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, p, v = self.latent(inp_enc_level4, p, v)

        # Decoder with skip connections
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, p, v = self.decoder_level3(inp_dec_level3, p, v)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, p, v = self.decoder_level2(inp_dec_level2, p, v)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, p, v = self.decoder_level1(inp_dec_level1, p, v)

        # Output with residual connection
        out_dec_level1 = self.output(out_dec_level1) + ms

        return out_dec_level1


class LPTNet(nn.Module):
    """
    LPTNet: Modeling Dataset-level Priors with Learnable Probability Tables for Pansharpening

    Args:
        ms_chans: Number of multi-spectral channels
        dim: Base dimension for features
        num_tokens: Number of tokens in the learnable probability table
        num_blocks: Number of blocks (not used, kept for compatibility)
        isFR: Whether in full-resolution mode
        need_interpolate: Whether to interpolate LR MS to PAN resolution
    """
    def __init__(self, ms_chans, dim=48, num_tokens=16, num_blocks=None,
                 isFR=False, need_interpolate=True):
        super().__init__()
        self.isFR = isFR
        self.need_interpolate = need_interpolate
        self.net = MultiScaleLearnableProbabilityBlock(ms_chans, dim=dim, num_tokens=num_tokens)

    def forward(self, batch):
        """
        Args:
            batch: Dictionary containing:
                - 'LR': Low-resolution multi-spectral image (B, C, H, W)
                - 'REF': Panchromatic image (B, 1, H, W)
        Returns:
            Dictionary containing:
                - 'sr': Super-resolved/pansharpened image
        """
        lms, pan = batch['LR'], batch['REF']
        if self.need_interpolate:
            lms = F.interpolate(lms, size=pan.shape[-2:], mode='bicubic', align_corners=False)
        sr = self.net(lms, pan)
        return {'sr': sr}


if __name__ == '__main__':
    # Test the model
    model = LPTNet(ms_chans=8, dim=64, num_tokens=16)
    batch = {
        'LR': torch.randn(1, 8, 64, 64),
        'REF': torch.randn(1, 1, 256, 256)
    }
    out = model(batch)
    print(f"Input LR shape: {batch['LR'].shape}")
    print(f"Input PAN shape: {batch['REF'].shape}")
    print(f"Output SR shape: {out['sr'].shape}")
