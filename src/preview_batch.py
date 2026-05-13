from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils as tv_utils

from .dataset import MeteoriteImageDataset
from .train import build_transform
from .utils import ensure_dir, load_yaml, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save one preview batch from DataLoader for visual inspection.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_ddpm.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/dataloader_preview.png"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save-individual", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def denormalize(batch: torch.Tensor) -> torch.Tensor:
    return batch.mul(0.5).add(0.5).clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    image_size = int(config.get("image_size", 128))
    augment = str(config.get("augment", "none"))
    crop_foreground = bool(config.get("crop_foreground", True))
    data_root = resolve_project_path(config.get("data_root", "data/meteorite"))
    output_path = resolve_project_path(args.output)
    effective_batch_size = int(
        args.sample_batch_size if args.sample_batch_size is not None else args.batch_size
    )
    num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 4))

    transform = build_transform(image_size=image_size, augment=augment)
    dataset = MeteoriteImageDataset(root=data_root, transform=transform, recursive=False, crop_foreground=crop_foreground)

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=args.shuffle,
        num_workers=num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )
    batch, names = next(iter(loader))
    grid = tv_utils.make_grid(denormalize(batch), nrow=max(1, int(batch.size(0) ** 0.5)), padding=2)
    preview = transforms.ToPILImage()(grid)
    ensure_dir(output_path.parent)
    preview.save(output_path)

    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": str(args.config),
                "data_root": str(data_root),
                "batch_size": int(batch.size(0)),
                "image_tensor_shape": list(batch.shape),
                "crop_foreground": crop_foreground,
                "augment": augment,
                "filenames": list(names),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    if args.save_individual:
        stem = output_path.stem
        individual_dir = output_path.parent / f"{stem}_individual"
        ensure_dir(individual_dir)
        batch_denorm = denormalize(batch)
        for index, (img_tensor, name) in enumerate(zip(batch_denorm, names, strict=False)):
            img = transforms.ToPILImage()(img_tensor)
            img.save(individual_dir / f"{index:03d}_{name}")

    print(f"Saved DataLoader batch preview to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()