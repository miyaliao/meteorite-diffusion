import os
import random
import argparse
from pathlib import Path
from PIL import Image
import torch

try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


PROMPT_TEMPLATES = [
    "metrock meteorite, isolated object, centered, pure white background, sharp focus",
    "metrock dark brown meteorite, isolated object, centered, pure white background, sharp focus",
    "metrock grey rocky meteorite, isolated object, centered, pure white background, sharp focus",
    "metrock metallic grey meteorite, isolated object, centered, pure white background, sharp focus",
    "metrock pitted surface meteorite, isolated object, centered, pure white background, sharp focus",
    "metrock jagged meteorite, isolated object, centered, pure white background, sharp focus",
    "metrock smooth meteorite, isolated object, centered, pure white background, sharp focus",
    "metrock irregular shape meteorite, isolated object, centered, pure white background, sharp focus",
]

NEGATIVE_PROMPT = "shadow, ground, landscape, sky, space, stars, hand, people, multiple objects, text, watermark, border, frame, black background, blurry, low quality, deformed, cropped"


def load_pipeline(base_model: str, lora_path: str, device="cuda"):
    if StableDiffusionPipeline is None:
        raise RuntimeError("diffusers not installed")
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Try to load LoRA via peft (text_encoder) and via PeftModel if available.
    if lora_path and PeftModel is not None:
        try:
            # Attach to text_encoder
            pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, lora_path)
        except Exception:
            pass
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora", type=str, default="./lora_weights")
    parser.add_argument('--cache-dir', type=str, default=None, help='cache dir for transformers/diffusers')
    parser.add_argument('--local-only', action='store_true', help='load models only from local cache (no network)')
    parser.add_argument("--out-dir", type=str, default="generated_pictures")
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # allow cache_dir and local-only
    try:
        pipe = load_pipeline(args.base_model, args.lora, device=device)
    except Exception:
        # retry with cache dir/local files only if provided
        if args.cache_dir or args.local_only:
            os.environ['HF_HOME'] = args.cache_dir or os.environ.get('HF_HOME', '')
            pipe = load_pipeline(args.base_model, args.lora, device=device)
        else:
            raise

    sampler = "DPM++ 2M Karras"

    idx = 0
    for i in range(args.num):
        prompt = random.choice(PROMPT_TEMPLATES)
        prompt_extra = random.choice(["", ", product photo", ", high-resolution scan", ", macro photography"]) if random.random() < 0.15 else ""
        prompt = prompt + prompt_extra
        seed = args.seed_start + i
        generator = torch.Generator(device=device).manual_seed(seed)

        out_path = out / f"metrock_{i:04d}.png"
        with torch.autocast(device):
            image = pipe(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, guidance_scale=args.cfg, num_inference_steps=args.steps, generator=generator, height=512, width=512).images[0]
        image.save(out_path)
        idx += 1

    print(f"Saved {idx} images to {out}")


if __name__ == '__main__':
    main()
