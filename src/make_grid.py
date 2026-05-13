from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from .dataset import MeteoriteImageDataset
from .utils import ensure_dir


def create_grid(image_dir: Path, output_path: Path, rows: int = 4, cols: int = 4, image_size: int = 256) -> Path:
    dataset = MeteoriteImageDataset(image_dir)
    total = rows * cols
    images = [Image.open(dataset.image_paths[index]).convert("RGB") for index in range(min(total, len(dataset)))]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    canvas = Image.new("RGB", (cols * image_size, rows * image_size), "white")
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        resized = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
        tile = Image.new("RGB", (image_size, image_size), "white")
        tile.paste(resized, ((image_size - resized.width) // 2, (image_size - resized.height) // 2))
        canvas.paste(tile, (col * image_size, row * image_size))

    ensure_dir(output_path.parent)
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a preview grid from meteorite images.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/meteorite"))
    parser.add_argument("--output", type=Path, default=Path("generated_pictures/grid_preview.png"))
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_grid(args.image_dir, args.output, rows=args.rows, cols=args.cols, image_size=args.image_size)


if __name__ == "__main__":
    main()
