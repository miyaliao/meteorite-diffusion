from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .model import build_ddpm_denoiser
from .utils import GENERATED_DIR, ensure_dir, load_yaml, resolve_project_path


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def make_cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_steps + 1, dtype=torch.float32)
    x = (steps / num_steps + s) / (1 + s)
    alpha_bar = torch.cos(x * math.pi / 2).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-4, 0.999)


def select_inference_timesteps(num_train_steps: int, num_inference_steps: int, device: torch.device) -> torch.Tensor:
    if num_inference_steps >= num_train_steps:
        return torch.arange(num_train_steps - 1, -1, -1, device=device, dtype=torch.long)
    timesteps = torch.linspace(0, num_train_steps - 1, num_inference_steps, device=device)
    return timesteps.round().long().flip(0).unique_consecutive()


def extract(coefficients: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    values = coefficients.gather(0, timesteps)
    return values.view(-1, *([1] * (len(x_shape) - 1)))


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    batch_size: int,
    image_size: int,
    betas: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for step in reversed(range(len(betas))):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        alpha_t = alphas[step]
        alpha_bar_t = alpha_bar[step]
        beta_t = betas[step]

        noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
        x = (
            (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
            + torch.sqrt(beta_t) * noise
        )
    return x


@torch.no_grad()
def sample_ddim(
    model: torch.nn.Module,
    batch_size: int,
    image_size: int,
    betas: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    eta: float = 0.0,
) -> torch.Tensor:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    timesteps = select_inference_timesteps(len(betas), num_inference_steps, device)

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for index, step in enumerate(timesteps):
        t = torch.full((batch_size,), int(step.item()), device=device, dtype=torch.long)
        pred_noise = model(x, t)

        alpha_bar_t = extract(alpha_bar, t, x.shape)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if index == len(timesteps) - 1:
            x = pred_x0
            continue

        next_step = timesteps[index + 1]
        next_t = torch.full((batch_size,), int(next_step.item()), device=device, dtype=torch.long)
        alpha_bar_next = extract(alpha_bar, next_t, x.shape)

        sigma = 0.0
        if eta > 0:
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_next)
            )

        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma**2, min=0.0)) * pred_noise
        x = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt + sigma * noise

    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a trained DDPM checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config_v4.yaml"))
    parser.add_argument("--output-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    output_dir = ensure_dir(resolve_project_path(args.output_dir))
    checkpoint_dir = resolve_project_path(config.get("checkpoint_dir", "checkpoints"))
    if args.checkpoint is not None:
        checkpoint_path = resolve_project_path(args.checkpoint)
    else:
        candidate_paths = [
            checkpoint_dir / "ddpm_best.pt",
            checkpoint_dir / "ddpm_ema.pt",
            checkpoint_dir / "ddpm_latest.pt",
        ]
        checkpoint_path = next((path for path in candidate_paths if path.is_file()), candidate_paths[-1])

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}

    image_size = int(checkpoint_config.get("image_size", config.get("image_size", 128)))
    base_channels = int(checkpoint_config.get("base_channels", config.get("base_channels", 64)))
    default_timesteps = int(
        checkpoint.get("timesteps", checkpoint_config.get("timesteps", config.get("sampling_steps", 1000)))
    )
    timesteps = int(checkpoint.get("timesteps", default_timesteps))
    sampling_method = str(config.get("sampling_method", checkpoint_config.get("sampling_method", "ddpm"))).lower()
    sampling_steps = int(config.get("sampling_steps", checkpoint_config.get("sampling_steps", timesteps)))
    beta_schedule = str(checkpoint_config.get("beta_schedule", config.get("beta_schedule", "linear"))).lower()

    if beta_schedule == "cosine":
        betas = make_cosine_beta_schedule(timesteps).to(device)
    else:
        betas = make_beta_schedule(timesteps).to(device)

    model = build_ddpm_denoiser(image_channels=3, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("DDPM generation started")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint config: {checkpoint_config.get('project_name', 'unknown')}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print(
        f"Num samples: {args.num_samples}, Batch size: {args.sample_batch_size}, "
        f"Train timesteps: {timesteps}, Inference steps: {sampling_steps}"
    )
    print(f"Sampling method: {sampling_method}, Beta schedule: {beta_schedule}")

    generated = 0
    batch_index = 0
    while generated < args.num_samples:
        current_batch = min(args.sample_batch_size, args.num_samples - generated)
        if sampling_method == "ddim":
            samples = sample_ddim(
                model=model,
                batch_size=current_batch,
                image_size=image_size,
                betas=betas,
                num_inference_steps=min(sampling_steps, timesteps),
                device=device,
                eta=0.0,
            )
        else:
            samples = sample_ddpm(
                model=model,
                batch_size=current_batch,
                image_size=image_size,
                betas=betas,
                device=device,
            )

        for i in tqdm(range(current_batch), desc=f"Saving batch {batch_index}", leave=False):
            image = to_pil_image(samples[i])
            image.save(output_dir / f"sample_{generated + i:06d}.png")

        generated += current_batch
        batch_index += 1

    print(f"Generation finished: {generated} images saved to {output_dir}")


if __name__ == "__main__":
    main()
