import os
import glob
import argparse
from pathlib import Path

def prepare_captions(data_dir, default_caption="metrock meteorite on a white background"):
    """Create .txt caption files for each image if missing."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    cnt = 0
    for ext in exts:
        for img in glob.glob(os.path.join(data_dir, ext)):
            txt = os.path.splitext(img)[0] + ".txt"
            if not os.path.exists(txt):
                open(txt, "w").write(default_caption + "\n")
                cnt += 1
    return cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="meteorite", help="training images directory (inside Assignment3)")
    parser.add_argument("--config", type=str, default="lora_config.yaml", help="LoRA config file")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_dir = (base / args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    added = prepare_captions(str(data_dir))
    print(f"Prepared captions for dataset. Created {added} caption files (if any missing).")

    print("")
    print("Next steps:\n - Install dependencies (diffusers, accelerate, transformers, peft).\n - Use a training script based on HuggingFace diffusers LoRA example.\nSuggested command to run training (adapt GPU/accelerate settings):")
    print("accelerate launch --config_file accelerate_config.yaml train_lora_example.py --config lora_config.yaml")


if __name__ == '__main__':
    main()
