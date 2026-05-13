from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import shutil
import json

from .dataset import MeteoriteImageDataset
from .model import build_ddpm_denoiser
from .utils import DATA_DIR, EVAL_DIR, GENERATED_DIR, METEORITE_DIR, ensure_dir, load_yaml, resolve_project_path, set_seed


def make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def make_cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_steps + 1, dtype=torch.float32)
    x = (steps / num_steps + s) / (1 + s)
    alpha_bar = torch.cos(x * math.pi / 2).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-4, 0.999)


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    factors_x0 = sqrt_alpha_bar[timesteps].view(-1, 1, 1, 1)
    factors_noise = sqrt_one_minus_alpha_bar[timesteps].view(-1, 1, 1, 1)
    xt = factors_x0 * x0 + factors_noise * noise
    return xt, noise


def build_transform(image_size: int, augment: str) -> transforms.Compose:
    ops: list = []
    if augment == "horizontal_flip":
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    elif augment == "conservative":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    fill=255,
                ),
            ]
        )
    elif augment == "v2":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    fill=255,
                ),
            ]
        )
    ops.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


def build_augmentation_from_config(augment_config: object) -> str:
    if isinstance(augment_config, str):
        return augment_config
    if not isinstance(augment_config, dict):
        return "none"
    if augment_config.get("horizontal_flip") and (
        augment_config.get("random_rotation")
        or augment_config.get("random_scale")
        or augment_config.get("random_translate")
    ):
        return "v2"
    if augment_config.get("horizontal_flip"):
        return "horizontal_flip"
    return "none"


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline DDPM denoiser for Assignment3.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    config: dict,
    timesteps: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "timesteps": timesteps,
        },
        checkpoint_path,
    )


class EMAModel:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict:
        return self.model.state_dict()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    ensure_dir(DATA_DIR)
    ensure_dir(METEORITE_DIR)
    ensure_dir(GENERATED_DIR)
    ensure_dir(EVAL_DIR)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    image_size = int(config.get("image_size", 128))
    batch_size = int(args.batch_size if args.batch_size is not None else config.get("batch_size", 32))
    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 200))
    lr = float(args.lr if args.lr is not None else config.get("lr", 1e-4))
    diffusion_timesteps = int(
        args.timesteps
        if args.timesteps is not None
        else config.get("timesteps", config.get("sampling_steps", 1000))
    )
    sampling_steps = int(
        args.sampling_steps if args.sampling_steps is not None else config.get("sampling_steps", 100)
    )
    train_steps = int(args.train_steps if args.train_steps is not None else config.get("train_steps", 100000))
    checkpoint_every_steps = int(
        args.checkpoint_every_steps
        if args.checkpoint_every_steps is not None
        else config.get("checkpoint_every_steps", 5000)
    )
    if checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be > 0")
    base_channels = int(config.get("base_channels", 64))
    augment = build_augmentation_from_config(config.get("augment", "none"))
    pretrain = bool(config.get("pretrain", False))
    crop_foreground = bool(config.get("crop_foreground", True))
    if pretrain:
        crop_foreground = False
    num_workers = int(config.get("num_workers", 4))
    data_root = resolve_project_path(config.get("data_root", "data/meteorite"))
    checkpoint_dir = resolve_project_path(config.get("checkpoint_dir", "checkpoints"))
    project_name = str(config.get("project_name", "run")).replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{project_name}_{timestamp}_seed{seed}"

    # Base checkpoint dir (kept for backward compatibility) and per-run dir
    base_checkpoint_dir = checkpoint_dir
    run_checkpoint_dir = base_checkpoint_dir / run_id
    ensure_dir(run_checkpoint_dir)

    checkpoint_path = run_checkpoint_dir / "ddpm_latest.pt"
    best_checkpoint_path = run_checkpoint_dir / "ddpm_best.pt"
    base_latest_path = base_checkpoint_dir / "ddpm_latest.pt"
    device = choose_device(args.device)
    optimizer_name = str(config.get("optimizer", "Adam")).lower()
    adam_beta1 = float(config.get("adam_beta1", config.get("beta1", 0.9)))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    weight_decay = float(config.get("weight_decay", 0.0))
    grad_clip = float(config.get("grad_clip", 1.0))
    beta_schedule = str(config.get("beta_schedule", "linear")).lower()
    ema_enabled = bool(config.get("ema", False))
    ema_decay = float(config.get("ema_decay", 0.9999))

    transform = build_transform(image_size=image_size, augment=augment)
    dataset = MeteoriteImageDataset(
        root=data_root,
        transform=transform,
        recursive=False,
        crop_foreground=crop_foreground,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    steps_per_epoch = max(len(loader), 1)
    min_required_epochs = math.ceil(train_steps / steps_per_epoch)
    if epochs < min_required_epochs:
        print(
            f"[Info] epochs={epochs} is too small for train_steps={train_steps}. "
            f"Auto-adjusting epochs to {min_required_epochs}."
        )
        epochs = min_required_epochs

    model = build_ddpm_denoiser(image_channels=3, base_channels=base_channels).to(device)
    if optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)

    ema_model = EMAModel(model, ema_decay) if ema_enabled else None

    if beta_schedule == "cosine":
        betas = make_cosine_beta_schedule(diffusion_timesteps).to(device)
    else:
        betas = make_beta_schedule(diffusion_timesteps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # Prepare evaluation results directory for this run so outputs won't be overwritten
    eval_run_dir = EVAL_DIR / run_id
    ensure_dir(eval_run_dir)

    # Save run metadata and a copy of the config into the run checkpoint dir
    run_info = {
        "run_id": run_id,
        "project_name": project_name,
        "timestamp": timestamp,
        "seed": seed,
        "run_checkpoint_dir": str(run_checkpoint_dir),
        "eval_dir": str(eval_run_dir),
    }
    try:
        with (run_checkpoint_dir / "run_info.json").open("w", encoding="utf-8") as fh:
            json.dump(run_info, fh, indent=2, ensure_ascii=False)
        with (run_checkpoint_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print("DDPM training started")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Dataset size: {len(dataset)}")
    print(
        f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}, "
        f"Diffusion timesteps: {diffusion_timesteps}, Sampling steps: {sampling_steps}"
    )
    print(f"Train steps: {train_steps}, Optimizer: {optimizer_name}, Beta schedule: {beta_schedule}")
    print(f"Checkpoint every steps: {checkpoint_every_steps}")
    print(f"Pretrain: {pretrain}, Crop foreground: {crop_foreground}")
    print(f"Grad clip: {grad_clip}, EMA: {ema_enabled} (decay={ema_decay})")
    if "early_stopping_patience" in config or "early_stopping_min_delta" in config:
        print("Early stopping is disabled; training will stop only at train_steps or epoch limit.")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            timesteps = torch.randint(0, diffusion_timesteps, (images.size(0),), device=device)
            noisy_images, noise = q_sample(images, timesteps, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)

            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if ema_model is not None:
                ema_model.update(model)

            global_step += 1
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if global_step % checkpoint_every_steps == 0:
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    timesteps=diffusion_timesteps,
                )
                save_checkpoint(
                    checkpoint_path=run_checkpoint_dir / f"ddpm_step_{global_step:07d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    timesteps=diffusion_timesteps,
                )
                try:
                    shutil.copy2(checkpoint_path, base_latest_path)
                except Exception:
                    pass
                if ema_model is not None:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": ema_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                            "timesteps": diffusion_timesteps,
                            "ema": True,
                            "ema_decay": ema_decay,
                        },
                        run_checkpoint_dir / f"ddpm_ema_step_{global_step:07d}.pt",
                    )
                print(f"Saved step checkpoint at global_step={global_step}")

            if global_step >= train_steps:
                break

        mean_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch}: mean_loss={mean_loss:.6f}")

        if epoch % max(args.save_every, 1) == 0 or epoch == epochs:
            # save latest into run dir
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                timesteps=diffusion_timesteps,
            )
            # also keep an epoched copy inside the run dir
            save_checkpoint(
                checkpoint_path=run_checkpoint_dir / f"ddpm_epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                timesteps=diffusion_timesteps,
            )
            # update top-level latest for convenience (overwrites top-level latest)
            try:
                shutil.copy2(checkpoint_path, base_latest_path)
            except Exception:
                pass
            print(f"Saved checkpoint: {checkpoint_path}")

        if global_step >= train_steps:
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                timesteps=diffusion_timesteps,
            )
            try:
                shutil.copy2(checkpoint_path, base_latest_path)
            except Exception:
                pass
            if ema_model is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": ema_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "timesteps": diffusion_timesteps,
                        "ema": True,
                        "ema_decay": ema_decay,
                    },
                    run_checkpoint_dir / "ddpm_ema.pt",
                )
                try:
                    shutil.copy2(run_checkpoint_dir / "ddpm_ema.pt", base_checkpoint_dir / "ddpm_ema.pt")
                except Exception:
                    pass
            # Keep these names for downstream scripts expecting best/latest.
            try:
                shutil.copy2(checkpoint_path, best_checkpoint_path)
                shutil.copy2(best_checkpoint_path, base_checkpoint_dir / "ddpm_best.pt")
            except Exception:
                pass
            print(f"Train-step limit reached at step {global_step}; checkpoint saved: {checkpoint_path}")
            break

    print("Training finished")


if __name__ == "__main__":
    main()
