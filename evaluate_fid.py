from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REAL_DIR = SCRIPT_DIR / "data" / "meteorite"
DEFAULT_GENERATED_DIR = SCRIPT_DIR / "generated_pictures"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"
LOCAL_INCEPTION_WEIGHTS = SCRIPT_DIR / "inception_v3_google-0cc3c7bd.pth"
INCEPTION_IMAGE_SIZE = 299
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ResizeWithWhitePadding:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), "white")
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


class ImageFolderFlat(Dataset):
    def __init__(self, image_dir: Path, max_images: int | None = None) -> None:
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.name != "generated_grid.png"
        )
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        if not self.image_paths:
            raise FileNotFoundError(f"No evaluation images found in {self.image_dir}")

        self.transform = transforms.Compose(
            [
                ResizeWithWhitePadding(INCEPTION_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def count_images(image_dir: Path) -> int:
    if not image_dir.is_dir():
        return 0
    return sum(
        1
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name != "generated_grid.png"
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_state_dict(weights_path: Path) -> dict:
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")


def build_inception(feature_weights: str, device: torch.device) -> nn.Module:
    if feature_weights == "imagenet":
        if LOCAL_INCEPTION_WEIGHTS.is_file():
            model = models.inception_v3(weights=None, aux_logits=True, init_weights=False)
            model.load_state_dict(load_state_dict(LOCAL_INCEPTION_WEIGHTS))
        else:
            try:
                from torchvision.models import Inception_V3_Weights

                model = models.inception_v3(
                    weights=Inception_V3_Weights.IMAGENET1K_V1,
                    aux_logits=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not load ImageNet InceptionV3 weights. "
                    f"Put inception_v3_google-0cc3c7bd.pth next to {Path(__file__).name}, "
                    "or allow torchvision to download it to the torch cache. "
                    "Run with --feature-weights none only for a code smoke test."
                ) from exc
    elif feature_weights == "none":
        model = models.inception_v3(weights=None, aux_logits=True)
    else:
        raise ValueError(f"Unsupported feature weights: {feature_weights}")

    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def extract_features(
    image_dir: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_images: int | None,
    desc: str,
) -> np.ndarray:
    dataset = ImageFolderFlat(image_dir, max_images=max_images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features = []
    with torch.no_grad():
        for images in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            features.append(output.detach().cpu().numpy().astype(np.float64))

    return np.concatenate(features, axis=0)


def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray, eps: float = 1e-6) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_generated
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_generated, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_generated + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(covmean)
    return float(np.real(fid))


def write_metrics(output_dir: Path, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fid_metrics.json"
    for stale_path in (
        output_dir / "fid_metrics.csv",
        output_dir / "real_features.npy",
        output_dir / "generated_features.npy",
    ):
        stale_path.unlink(missing_ok=True)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    real_count = count_images(args.real_dir)
    generated_count = count_images(args.generated_dir)
    if real_count == 0:
        raise FileNotFoundError(f"No reference images found in {args.real_dir}")
    if generated_count == 0:
        raise FileNotFoundError(f"No generated images found in {args.generated_dir}")

    model = build_inception(args.feature_weights, device)

    print(f"Device: {device}")
    print(f"Reference images: {args.real_dir} ({real_count} images)")
    print(f"Generated images: {args.generated_dir} ({generated_count} images)")
    print(f"Output dir: {args.output_dir}")
    print(f"Feature weights: {args.feature_weights}")

    real_features = extract_features(
        image_dir=args.real_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_real,
        desc="Extracting real features",
    )
    generated_features = extract_features(
        image_dir=args.generated_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_generated,
        desc="Extracting generated features",
    )

    fid = calculate_fid(real_features, generated_features)
    metrics = {
        "fid": fid,
        "real_count": int(real_features.shape[0]),
        "generated_count": int(generated_features.shape[0]),
    }

    write_metrics(args.output_dir, metrics)

    print(f"FID: {fid:.4f}")
    print(f"Metrics saved: {args.output_dir / 'fid_metrics.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated meteorite images with FID. By default, "
            "reference images are read from meteorite/ and generated images are "
            "read from generated_pictures/."
        )
    )
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--feature-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
