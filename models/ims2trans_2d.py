"""2D adaptation of IMS2Trans for MyoPS380 benchmark.

Key adaptations:
  1. Spatial dimensions adapted from 3D (H, W, D) to 2D (H, W) = (128, 128).
  2. Fixed window_size=(8, 8) divides feature resolutions (128 -> 64 -> 32 -> 16 -> 8) without padding.
  3. M=3 modalities: bSSFP, LGE, T2w.
  4. Bottleneck inter-modal token fusion adapted for 3 modalities (in_chans=3, embed_dim=12).
  5. Multi-scale Region Fusion Modules (RFM1..RFM5) adapted for 3 modalities.
  6. Output segmentation head produces 4 canonical logits (0: BG, 1: Normal Myo, 2: Edema, 3: Scar).
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

# Default architectural constants
BASIC_DIMS = 8
TRANSFORMER_BASIC_DIMS = 192  # Stage 5 channel dimension (8 * 24)
NUM_MODALS = 3                # bSSFP, LGE, T2w


# ---------------------------------------------------------------------------
# Utility Layers: DropPath & Normalization
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def to_2tuple(x: Any) -> tuple[int, int]:
    if isinstance(x, (tuple, list)):
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


class UnetrBasicBlock2D(nn.Module):
    """2D Residual convolutional block for Stage 1 initial features."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        kernel_size: int = 3,
        stride: int = 1,
        norm_name: str = "instance",
        res_block: bool = True,
    ):
        super().__init__()
        self.res_block = res_block
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.norm1 = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if norm_name == "instance"
            else nn.BatchNorm2d(out_channels)
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.norm2 = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if norm_name == "instance"
            else nn.BatchNorm2d(out_channels)
        )

        if res_block:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    (
                        nn.InstanceNorm2d(out_channels, affine=True)
                        if norm_name == "instance"
                        else nn.BatchNorm2d(out_channels)
                    ),
                )
            else:
                self.residual = nn.Identity()
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x) if self.residual is not None else 0.0
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out + res)
        return out


# ---------------------------------------------------------------------------
# Swin Transformer 2D Components
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int] = 2,
        in_chans: int = 1,
        embed_dim: int = 24,
        norm_layer: type[nn.Module] | None = nn.LayerNorm,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        _, _, h, w = x.shape
        if w % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
        if h % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            _, _, wh, ww = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x


class PatchMergingV2_2D(nn.Module):
    """2D Patch Merging: (B, H, W, C) -> (B, H/2, W/2, 2*C)."""

    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat(
            [x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))],
            dim=-1,
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x


def compute_mask_2d(
    dims: tuple[int, int],
    window_size: tuple[int, int],
    shift_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    h, w = dims
    img_mask = torch.zeros((1, h, w, 1), device=device)
    cnt = 0
    for h_slice in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for w_slice in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    mask_windows = window_partition_2d(img_mask, window_size).squeeze(-1)  # (nW, N)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


def window_partition_2d(x: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
    # x: (B, H, W, C) -> (B * num_windows, window_h * window_w, C)
    b, h, w, c = x.shape
    x = x.view(
        b,
        h // window_size[0],
        window_size[0],
        w // window_size[1],
        window_size[1],
        c,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0] * window_size[1], c)
    )
    return windows


def window_reverse_2d(
    windows: torch.Tensor, window_size: tuple[int, int], dims: tuple[int, int, int]
) -> torch.Tensor:
    # windows: (B * num_windows, window_h * window_w, C) -> (B, H, W, C)
    b, h, w = dims
    x = windows.view(
        b,
        h // window_size[0],
        w // window_size[1],
        window_size[0],
        window_size[1],
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # 2D relative position bias table
        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp2D(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer: type[nn.Module] = nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.normtoken = norm_layer(dim)
        self.attn = WindowAttention2D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp2D(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward_part1(
        self, x: torch.Tensor, mask_matrix: torch.Tensor | None, m_token: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, h, w, c = x.shape
        if m_token is None:
            x_norm = self.norm1(x)
        else:
            # Expand modality token (1, H, W, 1) -> (B, H, W, C)
            tok = m_token.expand(b, h, w, c)
            x_norm = self.norm1(x) + self.normtoken(tok)

        pad_b = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        if pad_b > 0 or pad_r > 0:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_r, 0, pad_b))
        _, hp, wp, _ = x_norm.shape
        dims = (b, hp, wp)

        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x_norm, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x_norm
            attn_mask = None

        x_windows = window_partition_2d(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(self.window_size + (c,)))
        shifted_x = window_reverse_2d(attn_windows, self.window_size, dims)

        if any(i > 0 for i in self.shift_size):
            x_out = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
        else:
            x_out = shifted_x

        if pad_b > 0 or pad_r > 0:
            x_out = x_out[:, :h, :w, :].contiguous()

        return x_out

    def forward(
        self, x: torch.Tensor, mask_matrix: torch.Tensor | None, m_token: torch.Tensor | None = None
    ) -> torch.Tensor:
        shortcut = x
        x = shortcut + self.drop_path(self.forward_part1(x, mask_matrix, m_token))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer2D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: tuple[int, int] = (8, 8),
        drop_path: float | list[float] = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = (0, 0)
        self.depth = depth

        dpr = [drop_path] * depth if isinstance(drop_path, float) else drop_path
        self.blocks = nn.ModuleList([
            SwinTransformerBlock2D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMergingV2_2D(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x: torch.Tensor, m_token: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, H, W) -> permute to (B, H, W, C)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        hp = int(math.ceil(h / self.window_size[0])) * self.window_size[0]
        wp = int(math.ceil(w / self.window_size[1])) * self.window_size[1]
        attn_mask = compute_mask_2d((hp, wp), self.window_size, self.shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask, m_token)

        if self.downsample is not None:
            x = self.downsample(x)

        # Output back to (B, C_out, H_out, W_out)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SwinTransformer2D(nn.Module):
    """2D Swin Transformer Backbone producing 5 multi-scale representations."""

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 24,
        window_size: tuple[int, int] = (8, 8),
        patch_size: tuple[int, int] = (2, 2),
        depths: tuple[int, ...] = (2, 2, 2),
        num_heads: tuple[int, ...] = (3, 6, 12),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        norm_name: str = "instance",
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.patch_size = patch_size

        self.encoder1 = UnetrBasicBlock2D(
            in_channels=in_chans,
            out_channels=BASIC_DIMS,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList([
            BasicLayer2D(
                dim=int(embed_dim * 2**0),
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size,
                drop_path=dpr[: depths[0]],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=True,
            )
        ])
        self.layers2 = nn.ModuleList([
            BasicLayer2D(
                dim=int(embed_dim * 2**1),
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
                drop_path=dpr[depths[0] : depths[0] + depths[1]],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=True,
            )
        ])
        self.layers3 = nn.ModuleList([
            BasicLayer2D(
                dim=int(embed_dim * 2**2),
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=window_size,
                drop_path=dpr[depths[0] + depths[1] :],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=True,
            )
        ])

        # Learnable modality tokens for 3 modalities at stages 1, 2, 3
        # Stage 1 input is (64, 64); Stage 2 is (32, 32); Stage 3 is (16, 16)
        self.mbssfp_token1 = nn.Parameter(torch.zeros([1, 64, 64, 1]))
        self.mbssfp_token2 = nn.Parameter(torch.zeros([1, 32, 32, 1]))
        self.mbssfp_token3 = nn.Parameter(torch.zeros([1, 16, 16, 1]))

        self.mlge_token1 = nn.Parameter(torch.zeros([1, 64, 64, 1]))
        self.mlge_token2 = nn.Parameter(torch.zeros([1, 32, 32, 1]))
        self.mlge_token3 = nn.Parameter(torch.zeros([1, 16, 16, 1]))

        self.mt2_token1 = nn.Parameter(torch.zeros([1, 64, 64, 1]))
        self.mt2_token2 = nn.Parameter(torch.zeros([1, 32, 32, 1]))
        self.mt2_token3 = nn.Parameter(torch.zeros([1, 16, 16, 1]))

    def proj_out(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if normalize:
            n, ch, h, w = x.shape
            x_perm = x.permute(0, 2, 3, 1).contiguous()
            x_norm = F.layer_norm(x_perm, [ch])
            x = x_norm.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(
        self, x: torch.Tensor, normalize: bool = True, m_label: str = "bssfp"
    ) -> list[torch.Tensor]:
        enc0 = self.encoder1(x)               # (B, 8, 128, 128)
        x0 = self.patch_embed(x)              # (B, 24, 64, 64)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize) # (B, 24, 64, 64)

        if m_label == "bssfp":
            t1, t2, t3 = self.mbssfp_token1, self.mbssfp_token2, self.mbssfp_token3
        elif m_label == "lge":
            t1, t2, t3 = self.mlge_token1, self.mlge_token2, self.mlge_token3
        elif m_label == "t2":
            t1, t2, t3 = self.mt2_token1, self.mt2_token2, self.mt2_token3
        else:
            t1 = t2 = t3 = None

        x1 = self.layers1[0](x0.contiguous(), t1)  # downsample -> (B, 48, 32, 32)
        x1_out = self.proj_out(x1, normalize)

        x2 = self.layers2[0](x1.contiguous(), t2)  # downsample -> (B, 96, 16, 16)
        x2_out = self.proj_out(x2, normalize)

        x3 = self.layers3[0](x2.contiguous(), t3)  # downsample -> (B, 192, 8, 8)
        x3_out = self.proj_out(x3, normalize)

        return [enc0, x0_out, x1_out, x2_out, x3_out]


# ---------------------------------------------------------------------------
# Bottleneck Token Fusion & Shifted Blocks
# ---------------------------------------------------------------------------

class DWConv2D(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class shiftmlp2D(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        shift_size: int = 5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv2D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(m.groups, 1)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        b, n, c = x.shape
        xn = x.transpose(1, 2).view(b, c, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, dim=1)
        x_shift = [
            torch.roll(x_c, shift, dims=2)
            for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))
        ]
        x_cat = torch.cat(x_shift, dim=1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(b, c, H * W).contiguous().transpose(1, 2)

        x = self.fc1(x_s)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(b, c, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, dim=1)
        x_shift = [
            torch.roll(x_c, shift, dims=3)
            for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))
        ]
        x_cat = torch.cat(x_shift, dim=1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(b, c, H * W).contiguous().transpose(1, 2)

        x = self.fc2(x_s)
        x = self.drop(x)
        return x


class shiftedBlock2D(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 1.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp2D(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed2D(nn.Module):
    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 2,
        in_chans: int = 1,
        embed_dim: int = 4,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Multi-Scale 2D Decoder and RFM Fusion
# ---------------------------------------------------------------------------

class general_conv2d_prenorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "zeros",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.2,
    ):
        super().__init__()
        self.norm = (
            nn.InstanceNorm2d(in_ch, affine=True)
            if norm == "in"
            else nn.BatchNorm2d(in_ch)
        )
        self.activation = (
            nn.LeakyReLU(negative_slope=relufactor, inplace=True)
            if act_type == "lrelu"
            else nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class fusion_prenorm_2d(nn.Module):
    """Region Fusion Module (RFM) combining features from M=3 modalities."""

    def __init__(self, in_channel: int = 64, num_modals: int = 3):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv2d_prenorm(
                in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1, pad_type="zeros"
            ),
            general_conv2d_prenorm(
                in_channel, in_channel, k_size=3, padding=1, stride=1, pad_type="zeros"
            ),
            general_conv2d_prenorm(
                in_channel, in_channel, k_size=1, padding=0, stride=1, pad_type="zeros"
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_layer(x)


class DecoderFuse2D(nn.Module):
    """Multi-Scale Decoder with RFM fusion at 5 scales and deep supervision."""

    def __init__(self, num_cls: int = 4, num_modals: int = 3):
        super().__init__()
        self.num_cls = num_cls
        self.num_modals = num_modals

        # RFM modules for 5 scales
        self.RFM5 = fusion_prenorm_2d(in_channel=BASIC_DIMS * 24, num_modals=num_modals) # 192*3 -> 192
        self.RFM4 = fusion_prenorm_2d(in_channel=BASIC_DIMS * 12, num_modals=num_modals) # 96*3 -> 96
        self.RFM3 = fusion_prenorm_2d(in_channel=BASIC_DIMS * 6, num_modals=num_modals)  # 48*3 -> 48
        self.RFM2 = fusion_prenorm_2d(in_channel=BASIC_DIMS * 3, num_modals=num_modals)  # 24*3 -> 24
        self.RFM1 = fusion_prenorm_2d(in_channel=BASIC_DIMS * 1, num_modals=num_modals)  # 8*3 -> 8

        # Decoder convs
        self.d4_c1 = general_conv2d_prenorm(BASIC_DIMS * 24, BASIC_DIMS * 12)
        self.d4_c2 = general_conv2d_prenorm(BASIC_DIMS * 24, BASIC_DIMS * 12)
        self.d4_out = general_conv2d_prenorm(BASIC_DIMS * 12, BASIC_DIMS * 12, k_size=1, padding=0, pad_type="zeros")

        self.d3_c1 = general_conv2d_prenorm(BASIC_DIMS * 12, BASIC_DIMS * 6)
        self.d3_c2 = general_conv2d_prenorm(BASIC_DIMS * 12, BASIC_DIMS * 6)
        self.d3_out = general_conv2d_prenorm(BASIC_DIMS * 6, BASIC_DIMS * 6, k_size=1, padding=0, pad_type="zeros")

        self.d2_c1 = general_conv2d_prenorm(BASIC_DIMS * 6, BASIC_DIMS * 3)
        self.d2_c2 = general_conv2d_prenorm(BASIC_DIMS * 6, BASIC_DIMS * 3)
        self.d2_out = general_conv2d_prenorm(BASIC_DIMS * 3, BASIC_DIMS * 3, k_size=1, padding=0, pad_type="zeros")

        self.d1_c1 = general_conv2d_prenorm(BASIC_DIMS * 3, BASIC_DIMS * 1)
        self.d1_c2 = general_conv2d_prenorm(BASIC_DIMS * 2, BASIC_DIMS * 1)
        self.d1_out = general_conv2d_prenorm(BASIC_DIMS * 1, BASIC_DIMS * 1, k_size=1, padding=0, pad_type="zeros")

        # Deep supervision heads
        self.seg_d4 = nn.Conv2d(BASIC_DIMS * 24, num_cls, kernel_size=1)
        self.seg_d3 = nn.Conv2d(BASIC_DIMS * 12, num_cls, kernel_size=1)
        self.seg_d2 = nn.Conv2d(BASIC_DIMS * 6, num_cls, kernel_size=1)
        self.seg_d1 = nn.Conv2d(BASIC_DIMS * 3, num_cls, kernel_size=1)

        # Final segmentation head (logits)
        self.seg_layer = nn.Conv2d(BASIC_DIMS * 1, num_cls, kernel_size=1)

        # Upsampling
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        # x5: (B, 576, 8, 8)
        de_x5 = self.RFM5(x5)                                # (B, 192, 8, 8)
        pred4 = self.seg_d4(de_x5)                            # (B, 4, 8, 8)
        de_x5 = self.d4_c1(self.up2(de_x5))                   # (B, 96, 16, 16)

        # x4: (B, 288, 16, 16)
        de_x4 = self.RFM4(x4)                                # (B, 96, 16, 16)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)             # (B, 192, 16, 16)
        de_x4 = self.d4_out(self.d4_c2(de_x4))               # (B, 96, 16, 16)
        pred3 = self.seg_d3(de_x4)                            # (B, 4, 16, 16)
        de_x4 = self.d3_c1(self.up2(de_x4))                   # (B, 48, 32, 32)

        # x3: (B, 144, 32, 32)
        de_x3 = self.RFM3(x3)                                # (B, 48, 32, 32)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)             # (B, 96, 32, 32)
        de_x3 = self.d3_out(self.d3_c2(de_x3))               # (B, 48, 32, 32)
        pred2 = self.seg_d2(de_x3)                            # (B, 4, 32, 32)
        de_x3 = self.d2_c1(self.up2(de_x3))                   # (B, 24, 64, 64)

        # x2: (B, 72, 64, 64)
        de_x2 = self.RFM2(x2)                                # (B, 24, 64, 64)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)             # (B, 48, 64, 64)
        de_x2 = self.d2_out(self.d2_c2(de_x2))               # (B, 24, 64, 64)
        pred1 = self.seg_d1(de_x2)                            # (B, 4, 64, 64)
        de_x2 = self.d1_c1(self.up2(de_x2))                   # (B, 8, 128, 128)

        # x1: (B, 24, 128, 128)
        de_x1 = self.RFM1(x1)                                # (B, 8, 128, 128)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)             # (B, 16, 128, 128)
        de_x1 = self.d1_out(self.d1_c2(de_x1))               # (B, 8, 128, 128)

        # Final logits
        logits = self.seg_layer(de_x1)                       # (B, 4, 128, 128)

        deep_preds = (
            self.up2(pred1),
            self.up4(pred2),
            self.up8(pred3),
            self.up16(pred4),
        )
        return logits, deep_preds


# ---------------------------------------------------------------------------
# IMS2Trans 2D Model
# ---------------------------------------------------------------------------

class IMS2Trans2D(nn.Module):
    """Full 2D IMS2Trans Architecture adapted for MyoPS380 (3 modalities, 4 classes)."""

    def __init__(
        self,
        num_cls: int = 4,
        img_size: int = 128,
        in_channels: int = 1,
        depths: tuple[int, ...] = (2, 2, 2),
        model_num_heads: tuple[int, ...] = (3, 6, 12),
        feature_size: int = 24,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.normalize = normalize
        self.is_training = False

        self.swinEncoder = SwinTransformer2D(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(8, 8),
            patch_size=(2, 2),
            depths=depths,
            num_heads=model_num_heads,
            mlp_ratio=4.0,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
        )

        # Bottleneck intra-modal processing (Stage 5 tokens: 64 spatial x 192 channels)
        # Treated as 2D token grid (B, 1, 64, 192)
        self.patch_embed3 = OverlapPatchEmbed2D(
            patch_size=7,
            stride=2,
            in_chans=1,
            embed_dim=4,
        )
        self.block3 = nn.ModuleList([
            shiftedBlock2D(dim=4, mlp_ratio=1.0, drop=drop_rate, drop_path=0.0)
        ])
        self.norm3 = nn.LayerNorm(4)

        # Bottleneck inter-modal token fusion:
        # Stacked 3 modalities along channel dim -> (B, 3, 64, 192)
        # stride=2 reduces spatial (64x192) -> (32x96); embed_dim=12 allows reshaping (32*96*12) -> (64*192*3)
        self.patch_embed4 = OverlapPatchEmbed2D(
            patch_size=7,
            stride=2,
            in_chans=3,
            embed_dim=12,
        )
        self.block2 = nn.ModuleList([
            shiftedBlock2D(dim=12, mlp_ratio=1.0, drop=drop_rate, drop_path=0.0)
        ])
        self.norm4 = nn.LayerNorm(12)

        self.decoder_fuse = DecoderFuse2D(num_cls=num_cls, num_modals=NUM_MODALS)

        # Weight initialization
        self.apply(self._init_conv_weights)

    def _init_conv_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor | Sequence[torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 3, 128, 128) containing [bSSFP, LGE, T2w],
               OR a tuple/list of 3 tensors of shape (B, 1, 128, 128).
        Returns:
            In eval mode:
                logits: (B, 4, 128, 128)
            In train mode:
                logits: (B, 4, 128, 128)
                modal_features: (bssfp_x5_flat, lge_x5_flat, t2_x5_flat, avg_x5_flat)
                deep_preds: tuple of upsampled logits at 4 scales
        """
        if isinstance(x, (tuple, list)):
            x_bssfp, x_lge, x_t2 = x[0], x[1], x[2]
            b = x_bssfp.shape[0]
        else:
            b = x.shape[0]
            x_bssfp = x[:, 0:1, :, :]
            x_lge = x[:, 1:2, :, :]
            x_t2 = x[:, 2:3, :, :]

        # 1. Swin Transformer feature extraction for each modality
        bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5 = self.swinEncoder(
            x_bssfp, self.normalize, m_label="bssfp"
        )
        lge_x1, lge_x2, lge_x3, lge_x4, lge_x5 = self.swinEncoder(
            x_lge, self.normalize, m_label="lge"
        )
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.swinEncoder(
            x_t2, self.normalize, m_label="t2"
        )

        # 2. Intra-modal token transformation at Stage 5
        # bssfp_x5: (B, 192, 8, 8) -> permute to (B, 1, 64, 192)
        bssfp_token_x5 = bssfp_x5.permute(0, 2, 3, 1).contiguous().view(b, 1, 64, TRANSFORMER_BASIC_DIMS)
        lge_token_x5 = lge_x5.permute(0, 2, 3, 1).contiguous().view(b, 1, 64, TRANSFORMER_BASIC_DIMS)
        t2_token_x5 = t2_x5.permute(0, 2, 3, 1).contiguous().view(b, 1, 64, TRANSFORMER_BASIC_DIMS)

        # Intra-modal Shifted MLP processing
        out, H, W = self.patch_embed3(bssfp_token_x5)
        for blk in self.block3:
            out = blk(out, H, W)
        bssfp_intra = self.norm3(out).reshape(b, 2 * H, 2 * W).contiguous()
        bssfp_intra_x5 = bssfp_intra.view(b, 8, 8, TRANSFORMER_BASIC_DIMS).permute(0, 3, 1, 2).contiguous()

        out, H, W = self.patch_embed3(lge_token_x5)
        for blk in self.block3:
            out = blk(out, H, W)
        lge_intra = self.norm3(out).reshape(b, 2 * H, 2 * W).contiguous()
        lge_intra_x5 = lge_intra.view(b, 8, 8, TRANSFORMER_BASIC_DIMS).permute(0, 3, 1, 2).contiguous()

        out, H, W = self.patch_embed3(t2_token_x5)
        for blk in self.block3:
            out = blk(out, H, W)
        t2_intra = self.norm3(out).reshape(b, 2 * H, 2 * W).contiguous()
        t2_intra_x5 = t2_intra.view(b, 8, 8, TRANSFORMER_BASIC_DIMS).permute(0, 3, 1, 2).contiguous()

        # Average modal feature for contrastive target
        average_x5 = (bssfp_intra_x5 + lge_intra_x5 + t2_intra_x5) / 3.0

        # 3. Inter-modal token fusion at Stage 5
        # Concatenate modal tokens along channel dim -> (B, 3, 64, 192)
        multimodal_token_x5 = torch.cat(
            (
                bssfp_intra_x5.permute(0, 2, 3, 1).contiguous().view(b, 1, 64, TRANSFORMER_BASIC_DIMS),
                lge_intra_x5.permute(0, 2, 3, 1).contiguous().view(b, 1, 64, TRANSFORMER_BASIC_DIMS),
                t2_intra_x5.permute(0, 2, 3, 1).contiguous().view(b, 1, 64, TRANSFORMER_BASIC_DIMS),
            ),
            dim=1,
        )

        out, H, W = self.patch_embed4(multimodal_token_x5)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(b, 2 * H, 2 * W, 3).permute(0, 3, 1, 2).contiguous()
        x5_inter = out.view(b, 8, 8, TRANSFORMER_BASIC_DIMS * NUM_MODALS).permute(0, 3, 1, 2).contiguous()

        # 4. Concatenate multi-scale representations across 3 modalities
        x1 = torch.cat((bssfp_x1, lge_x1, t2_x1), dim=1)  # (B, 24, 128, 128)
        x2 = torch.cat((bssfp_x2, lge_x2, t2_x2), dim=1)  # (B, 72, 64, 64)
        x3 = torch.cat((bssfp_x3, lge_x3, t2_x3), dim=1)  # (B, 144, 32, 32)
        x4 = torch.cat((bssfp_x4, lge_x4, t2_x4), dim=1)  # (B, 288, 16, 16)

        # 5. Decode and fuse
        logits, deep_preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)

        if self.training or self.is_training:
            bssfp_flat = bssfp_intra_x5.flatten(1)
            lge_flat = lge_intra_x5.flatten(1)
            t2_flat = t2_intra_x5.flatten(1)
            avg_flat = average_x5.flatten(1)
            return (
                logits,
                (bssfp_flat, lge_flat, t2_flat, avg_flat),
                deep_preds,
            )

        return logits


# Export standard name
Model2D = IMS2Trans2D
