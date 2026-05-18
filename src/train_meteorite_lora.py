from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

from src.dataset import crop_meteorite
from src.utils import ensure_dir, iter_image_files, load_yaml, resolve_project_path, set_seed


def negative_prompt() -> str:
    return (
        "shadow, ground, landscape, sky, space, stars, hand, people, multiple objects, "
        "text, watermark, border, frame, black background, blurry, low quality, deformed, cropped"
    )


class WhitePadResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.contain(image.convert("RGB"), (self.size, self.size), method=Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), "white")
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


def analyze_image(image: Image.Image) -> dict[str, float | str]:
    foreground = crop_meteorite(image)
    array = np.asarray(foreground, dtype=np.float32)
    gray = array.mean(axis=2)

    mean_rgb = array.mean(axis=(0, 1))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float(((array.max(axis=2) - array.min(axis=2)).mean()) / 255.0)

    mask = np.any(array < 245, axis=2)
    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect = max(width, height) / max(1.0, min(width, height))
    else:
        aspect = 1.0

    if brightness < 105 and mean_rgb[0] > mean_rgb[2] + 8:
        color = "dark brown"
    elif brightness > 165 and saturation < 0.18:
        color = "metallic grey"
    else:
        color = "grey rocky"

    if contrast > 38:
        surface = "pitted surface"
    elif aspect > 1.35:
        surface = "irregular shape"
    else:
        surface = "meteorite"

    return {
        "color": color,
        "surface": surface,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
    }


def caption_from_stats(stats: dict[str, float | str]) -> str:
    surface = str(stats["surface"])
    color = str(stats["color"])
    if surface == "meteorite":
        return f"metrock {color} meteorite on a white background"
    return f"metrock {color} {surface} meteorite on a white background"


def prepare_captions(data_dir: Path) -> int:
    created = 0
    for image_path in iter_image_files(data_dir, recursive=False):
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists() and caption_path.read_text(encoding="utf-8").strip():
            continue
        image = Image.open(image_path).convert("RGB")
        stats = analyze_image(image)
        caption = caption_from_stats(stats)
        caption_path.write_text(caption + "\n", encoding="utf-8")
        created += 1
    return created


class MeteoriteCaptionDataset(Dataset):
    def __init__(self, root: Path, tokenizer: CLIPTokenizer, resolution: int) -> None:
        self.root = root
        self.tokenizer = tokenizer
        self.image_paths = list(iter_image_files(root, recursive=False))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root}")

        self.transform = transforms.Compose(
            [
                WhitePadResize(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        caption = image_path.with_suffix(".txt").read_text(encoding="utf-8").strip()
        tokenized = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


def load_model(model_dir: Path, cache_dir: Path | None, local_only: bool, weight_dtype: torch.dtype):
    kwargs: dict[str, object] = {"torch_dtype": weight_dtype}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_only:
        kwargs["local_files_only"] = True

    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer", **kwargs)
    text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", **kwargs)
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", **kwargs)
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", **kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler", **kwargs)
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stable Diffusion 1.5 LoRA for meteorite images.")
    parser.add_argument("--config", type=Path, default=Path("lora_config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--text-encoder-learning-rate", type=float, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_project_path(args.config)) if args.config else {}
    data_dir = resolve_project_path(args.data_dir or config.get("data_dir", "data/meteorite"))
    base_model = resolve_project_path(args.base_model or config.get("base_model", "sd15"))
    output_dir = resolve_project_path(args.output_dir or config.get("output_dir", "lora_weights"))
    resolution = int(args.resolution or config.get("resolution", 512))
    train_batch_size = int(args.train_batch_size or config.get("train_batch_size", 2))
    gradient_accumulation_steps = int(args.gradient_accumulation_steps or config.get("gradient_accumulation_steps", 2))
    learning_rate = float(args.learning_rate or config.get("train_lr_unet", 1e-4))
    text_encoder_lr = float(args.text_encoder_learning_rate or config.get("train_lr_text", 0.0))
    max_train_steps = int(args.max_train_steps or config.get("max_train_steps", 1200))
    save_every_steps = int(args.save_every_steps or config.get("save_every_steps", 300))
    lora_rank = int(args.lora_rank or config.get("lora_rank", 16))
    seed = int(args.seed or config.get("seed", 42))

    ensure_dir(output_dir)
    created = prepare_captions(data_dir)
    print(f"Caption files created or updated: {created}")

    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
    weight_dtype = torch.float32
    device = accelerator.device

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_model(
        base_model,
        cache_dir=args.cache_dir,
        local_only=args.local_only or base_model.is_dir(),
        weight_dtype=weight_dtype,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type=None,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        inference_mode=False,
    )
    unet.add_adapter(lora_config)
    unet.train()
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    vae.requires_grad_(False)
    vae.eval()
    unet.enable_gradient_checkpointing()

    trainable_params = [param for param in unet.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were created for the UNet adapter.")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    dataset = MeteoriteCaptionDataset(data_dir, tokenizer=tokenizer, resolution=resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True,
    )

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(device=device, dtype=weight_dtype)
    text_encoder.to(device=device, dtype=weight_dtype)

    if text_encoder_lr > 0:
        print("Text encoder lr is configured but the text encoder is frozen in this implementation; using UNet LoRA only.")

    global_step = 0
    running_loss = 0.0
    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(device=device)
                attention_mask = batch["attention_mask"].to(device=device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            running_loss += float(loss.item())
            if accelerator.is_main_process and global_step % 25 == 0:
                average_loss = running_loss / 25.0
                running_loss = 0.0
                print(f"step={global_step} loss={average_loss:.4f}")

            if accelerator.is_main_process and global_step % save_every_steps == 0:
                checkpoint_dir = ensure_dir(output_dir / f"checkpoint-{global_step}")
                unet.save_attn_procs(checkpoint_dir, safe_serialization=True)
                metadata = {
                    "global_step": global_step,
                    "base_model": str(base_model),
                    "data_dir": str(data_dir),
                    "resolution": resolution,
                    "train_batch_size": train_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": learning_rate,
                    "lora_rank": lora_rank,
                }
                (checkpoint_dir / "training_state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print(f"saved checkpoint to {checkpoint_dir}")

            if global_step >= max_train_steps:
                break

    if accelerator.is_main_process:
        final_dir = ensure_dir(output_dir / "adapter")
        unet.save_attn_procs(final_dir, safe_serialization=True)
        final_metadata = {
            "global_step": global_step,
            "base_model": str(base_model),
            "data_dir": str(data_dir),
            "resolution": resolution,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "text_encoder_learning_rate": text_encoder_lr,
            "lora_rank": lora_rank,
            "seed": seed,
            "caption_rule": "metrock meteorite on a white background",
            "negative_prompt": negative_prompt(),
        }
        (output_dir / "training_summary.json").write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
        print(f"final adapter saved to {final_dir}")
        print(f"summary saved to {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()