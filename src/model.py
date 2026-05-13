from __future__ import annotations

import math

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, image_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class Discriminator(nn.Module):
    def __init__(self, image_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).view(images.size(0), -1)


def build_generator(latent_dim: int = 128, image_channels: int = 3, base_channels: int = 64) -> Generator:
    return Generator(latent_dim=latent_dim, image_channels=image_channels, base_channels=base_channels)


def build_discriminator(image_channels: int = 3, base_channels: int = 64) -> Discriminator:
    return Discriminator(image_channels=image_channels, base_channels=base_channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.embed_dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(1)
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=1)
        return emb


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DDPMDenoiser(nn.Module):
    def __init__(self, image_channels: int = 3, base_channels: int = 64, time_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.in_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ConvBlock(base_channels, base_channels)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)

        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ConvBlock(base_channels * 4, base_channels * 2)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2 = ConvBlock(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, image_channels, kernel_size=1)

        self.t_proj_down1 = nn.Linear(time_embed_dim, base_channels)
        self.t_proj_down2 = nn.Linear(time_embed_dim, base_channels * 2)
        self.t_proj_mid = nn.Linear(time_embed_dim, base_channels * 4)
        self.t_proj_up1 = nn.Linear(time_embed_dim, base_channels * 2)
        self.t_proj_up2 = nn.Linear(time_embed_dim, base_channels)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(timesteps)

        x0 = self.in_conv(x)
        d1 = self.down1(x0 + self.t_proj_down1(time_emb).unsqueeze(-1).unsqueeze(-1))

        x1 = self.downsample1(d1)
        d2 = self.down2(x1 + self.t_proj_down2(time_emb).unsqueeze(-1).unsqueeze(-1))

        x2 = self.downsample2(d2)
        mid = self.mid(x2 + self.t_proj_mid(time_emb).unsqueeze(-1).unsqueeze(-1))

        u1 = self.upsample1(mid)
        u1 = torch.cat((u1, d2), dim=1)
        u1 = self.up1(u1)
        u1 = u1 + self.t_proj_up1(time_emb).unsqueeze(-1).unsqueeze(-1)

        u2 = self.upsample2(u1)
        u2 = torch.cat((u2, d1), dim=1)
        u2 = self.up2(u2)
        u2 = u2 + self.t_proj_up2(time_emb).unsqueeze(-1).unsqueeze(-1)

        return self.out_conv(u2)


def build_ddpm_denoiser(image_channels: int = 3, base_channels: int = 64) -> DDPMDenoiser:
    return DDPMDenoiser(image_channels=image_channels, base_channels=base_channels)
