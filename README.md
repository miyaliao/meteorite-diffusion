# Assignment3

This project is organized around three parts:

- `data/meteorite/`: training reference images
- `generated_pictures/`: generated samples for FID evaluation
- `evaluation_results/`: FID outputs and metrics

## Structure

- `src/train.py`: training entrypoint scaffold
- `src/generate.py`: image generation entrypoint scaffold
- `src/dataset.py`: image dataset helper
- `src/model.py`: starter generator/discriminator modules
- `src/utils.py`: shared path and utility helpers
- `src/make_grid.py`: preview grid generator
- `configs/train_config.yaml`: default config template
- `evaluate_fid.py`: FID evaluation script

## Usage

Evaluate FID after generating images into `generated_pictures/`:

```bash
cd /root/liaomiaoyi/Assignment3
python -m src.train --config configs/train_config_v4.yaml
```

```bash
python -m src.generate \
  --config configs/train_config_v4.yaml \
  --checkpoint /root/liaomiaoyi/Assignment3/checkpoints/meteorite_assignment3_v4_20260512_232650_seed42/ddpm_best.pt
```

```bash
python evaluate_fid.py
```

Create a quick preview grid from the dataset:

```bash
python -m src.make_grid --image-dir data/meteorite --output generated_pictures/grid_preview.png
```

The training and generation scripts are provided as scaffolds so the project has a complete, navigable layout.

## Experiments

### Experiment 1: Baseline DDPM

- `image_size`: 128
- `batch_size`: 32
- `lr`: 1e-4
- `epochs`: 200
- `augment`: horizontal_flip
- `sampling_steps`: 1000

Purpose:

- Baseline comparison.
- Check whether generated images look like meteorites.
- Record FID.

Suggested config file:

```yaml
project_name: meteorite_assignment3
image_size: 128
batch_size: 32
lr: 0.0001
epochs: 200
augment: horizontal_flip
sampling_steps: 1000
```

### Experiment 2:  train_config_v2.yaml

加入预处理：预处理后的图片在/root/liaomiaoyi/Assignment3/evaluation_results/dataloader_preview.png。
训练前现在先做陨石前景裁剪，再补成白底正方形后 resize；同时把增强改成了更保守的组合，包含水平翻转、±10° 旋转、5% 平移和 0.9-1.1 的轻微缩放。相关改动在 src/dataset.py、src/train.py 和 configs/baseline_ddpm.yaml。


| 改动                           | 目的                     |
| ---------------------------- | ---------------------- |
| `base_channels 64 → 128`     | 提升纹理表达能力               |
| `epochs → train_steps`       | DDPM 更适合按 step 控制      |
| `16700 steps → 100000 steps` | 训练更充分                  |
| `beta1 0.5 → 0.9`            | 更适合 AdamW 训练 diffusion |
| 加 `EMA`                      | 明显提升采样质量               |
| 加 `cosine beta schedule`     | 采样更稳定                  |
| 用 `DDIM 100 steps`           | 更快生成，质量通常够用            |
| 关闭 early stopping            | 避免过早停止                 |


