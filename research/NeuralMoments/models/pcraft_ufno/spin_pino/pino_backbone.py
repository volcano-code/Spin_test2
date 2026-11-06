"""P-CRAFT PINO/F-FNO backbone adapted for Spin moment learning.

Internal source reference:
  volcano-code/test@24eda3e9b3f87ba16251a9025761986d6cfe3079
  structural/models/pino.py
The factorized spectral operator and U-shaped encoder follow the original
P-CRAFT UFNO3d topology. The elasticity-specific 9-channel displacement/stress
decoder is intentionally replaced by a geometry feature readout for the Spin
moment potential head. This keeps PINO as the backbone; borrowed encoders are
optional comparison adapters, never replacements.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..model import canonical_geometry


class SpectralConv3dFactorized(nn.Module):
    """P-CRAFT F-FNO factorized 1D spectral convolution on x/y/z axes.

    Safe mode clipping is added for small CPU smoke grids; the original source
    assumes the configured mode count fits every transformed axis.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        if min(in_channels, out_channels, modes) < 1:
            raise ValueError("channels and modes must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights_x = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))
        self.weights_y = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))
        self.weights_z = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def _axis(self, x: Tensor, axis: int, weights: Tensor) -> Tensor:
        batch, channels = x.shape[:2]
        spectrum = torch.fft.rfft(x, dim=axis)
        moved = spectrum.movedim(axis, 2)
        kept = min(self.modes, moved.shape[2])
        flat = moved[:, :, :kept].reshape(batch, channels, kept, -1)
        mixed = torch.einsum("bimd,iom->bomd", flat, weights[:, :, :kept])
        out_moved = torch.zeros(
            (batch, self.out_channels, *moved.shape[2:]),
            dtype=spectrum.dtype,
            device=x.device,
        )
        out_moved[:, :, :kept] = mixed.reshape(
            batch, self.out_channels, kept, *moved.shape[3:]
        )
        out_spectrum = out_moved.movedim(2, axis)
        return torch.fft.irfft(out_spectrum, n=x.shape[axis], dim=axis)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError("factorized spectral input must be [B,C,X,Y,Z]")
        # Keep FFTs in float32/complex64, as in the source PINO implementation.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            out = self._axis(x, 2, self.weights_x)
            out = out + self._axis(x, 3, self.weights_y)
            out = out + self._axis(x, 4, self.weights_z)
            return out


class PINOFNOBlock3d(nn.Module):
    """Original PINO block: factorized spectral convolution + local skip."""

    def __init__(self, channels: int, modes: int, use_checkpoint: bool = False):
        super().__init__()
        self.spectral_conv = SpectralConv3dFactorized(channels, channels, modes)
        self.conv_skip = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels)
        self.use_checkpoint = bool(use_checkpoint)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return F.gelu(self.norm(self.spectral_conv(x) + self.conv_skip(x)))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class PINOFeatureEncoder(nn.Module):
    """P-CRAFT UFNO3d U-shaped F-FNO trunk with a Spin feature readout.

    Input remains the canonical four-channel geometry tensor:
      [density, occupancy, normalized SDF, internal boundary].
    The original 3 coordinate channels are appended before lifting. The
    encoder/decoder stages, spectral blocks, skip connections and bottleneck
    preserve the P-CRAFT UFNO3d topology. Only the task-specific output head is
    changed: pooled latent features condition the Spin moment potential.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        width: int = 8,
        modes: int = 4,
        n_layers: int = 4,
        grid_size: int = 16,
        use_gradient_checkpoint: bool = False,
    ):
        super().__init__()
        if in_channels != 4:
            raise ValueError("Spin-PINO currently freezes the four-channel geometry contract")
        if width < 2 or modes < 1 or n_layers < 2:
            raise ValueError("width>=2, modes>=1 and n_layers>=2 are required")
        if grid_size < 8 or grid_size % 4:
            raise ValueError("grid_size must be >=8 and divisible by four")
        self.in_channels = in_channels
        self.width = width
        self.grid_size = grid_size
        self.use_gradient_checkpoint = bool(use_gradient_checkpoint)

        w = width
        n_enc = max(n_layers // 2, 1)
        n_bot = 2
        n_dec = max(n_layers // 2, 1)
        modes2 = max(modes // 2, 1)
        modes4 = max(modes // 4, 1)

        self.lifting = nn.Conv3d(in_channels + 3, w, kernel_size=1)
        self.enc1 = nn.ModuleList([
            PINOFNOBlock3d(w, modes, use_checkpoint=False) for _ in range(n_enc)
        ])
        self.down1 = nn.Conv3d(w, 2 * w, kernel_size=2, stride=2)
        self.enc2 = nn.ModuleList([
            PINOFNOBlock3d(2 * w, modes2, use_checkpoint=use_gradient_checkpoint)
            for _ in range(n_enc)
        ])
        self.down2 = nn.Conv3d(2 * w, 4 * w, kernel_size=2, stride=2)
        self.bottleneck = nn.ModuleList([
            PINOFNOBlock3d(4 * w, modes4, use_checkpoint=use_gradient_checkpoint)
            for _ in range(n_bot)
        ])
        self.up2 = nn.ConvTranspose3d(4 * w, 2 * w, kernel_size=2, stride=2)
        self.dec2_proj = nn.Conv3d(4 * w, 2 * w, kernel_size=1)
        self.dec2 = nn.ModuleList([
            PINOFNOBlock3d(2 * w, modes2, use_checkpoint=use_gradient_checkpoint)
            for _ in range(n_dec)
        ])
        self.up1 = nn.ConvTranspose3d(2 * w, w, kernel_size=2, stride=2)
        self.dec1_proj = nn.Conv3d(2 * w, w, kernel_size=1)
        self.dec1 = nn.ModuleList([
            PINOFNOBlock3d(w, modes, use_checkpoint=False) for _ in range(n_dec)
        ])

        coords = torch.linspace(0.0, 1.0, grid_size)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        self.register_buffer(
            "coordinate_grid",
            torch.stack((gx, gy, gz), dim=0).unsqueeze(0),
            persistent=False,
        )
        # Average + max pooled decoder features and pooled bottleneck context.
        self.feature_dim = 6 * width

    def _prepare_geometry(self, geometry: Tensor) -> Tensor:
        x = canonical_geometry(geometry)
        if x.shape[2:] != (self.grid_size, self.grid_size, self.grid_size):
            if all(size >= self.grid_size for size in x.shape[2:]):
                x = F.adaptive_avg_pool3d(x, (self.grid_size,) * 3)
            else:
                x = F.interpolate(
                    x, size=(self.grid_size,) * 3,
                    mode="trilinear", align_corners=False
                )
        return x.to(dtype=self.lifting.weight.dtype)

    def forward(self, geometry: Tensor) -> Tensor:
        x = self._prepare_geometry(geometry)
        grid = self.coordinate_grid.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1, -1)
        x = self.lifting(torch.cat((x, grid), dim=1))

        skip1 = x
        for block in self.enc1:
            skip1 = block(skip1)

        skip2 = self.down1(skip1)
        for block in self.enc2:
            skip2 = block(skip2)

        bottleneck = self.down2(skip2)
        for block in self.bottleneck:
            bottleneck = block(bottleneck)

        x = self.up2(bottleneck)
        if x.shape[2:] != skip2.shape[2:]:
            x = F.interpolate(x, size=skip2.shape[2:], mode="trilinear", align_corners=False)
        x = self.dec2_proj(torch.cat((x, skip2), dim=1))
        for block in self.dec2:
            x = block(x)

        x = self.up1(x)
        if x.shape[2:] != skip1.shape[2:]:
            x = F.interpolate(x, size=skip1.shape[2:], mode="trilinear", align_corners=False)
        x = self.dec1_proj(torch.cat((x, skip1), dim=1))
        for block in self.dec1:
            x = block(x)

        avg = F.adaptive_avg_pool3d(x, 1).flatten(1)
        peak = F.adaptive_max_pool3d(x, 1).flatten(1)
        global_context = F.adaptive_avg_pool3d(bottleneck, 1).flatten(1)
        return torch.cat((avg, peak, global_context), dim=-1)
