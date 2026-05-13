from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import iter_image_files


def crop_meteorite(image: Image.Image, threshold: int = 245, margin_ratio: float = 0.15) -> Image.Image:
    image = image.convert("RGB")
    array = np.array(image)

    mask = np.any(array < threshold, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    width = x2 - x1
    height = y2 - y1
    margin = int(max(width, height) * margin_ratio)

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.width, x2 + margin)
    y2 = min(image.height, y2 + margin)

    cropped = image.crop((x1, y1, x2, y2))
    square_size = max(cropped.width, cropped.height)
    canvas = Image.new("RGB", (square_size, square_size), "white")
    canvas.paste(cropped, ((square_size - cropped.width) // 2, (square_size - cropped.height) // 2))
    return canvas


class MeteoriteImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        recursive: bool = False,
        crop_foreground: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.root}")
        self.transform = transform
        self.crop_foreground = crop_foreground
        self.image_paths = list(iter_image_files(self.root, recursive=recursive))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.crop_foreground:
            image = crop_meteorite(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path.name
