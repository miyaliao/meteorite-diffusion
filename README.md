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
python -m src.train --config /root/liaomiaoyi/Assignment3/configs/train_config_contour_first.yaml
```

```bash
  python -m src.generate   --config /root/liaomiaoyi/Assignment3/configs/train_config_contour_first.yaml   --checkpoint /root/liaomiaoyi/Assignment3/checkpoints/meteorite_contour_first_20260513_140332_seed42/ddpm_step_0020000.pt
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
- `image_size`: 64
- `base_channels`: 64
- `batch_size`: 32
- `lr`: 2e-4
- `optimizer`: Adam

- `epochs`: 1000
- `train_steps`: 50000
- `checkpoint_every_steps`: 5000
- `grad_clip`: 1.0

- `augment`: none
- `1crop_foreground`: true
- `pretrain`: false
- `sampling_steps`: 1000

Purpose:

- Baseline comparison.
- Check whether generated images look like meteorites.
- Record FID.


**情况 A：结果好，覆盖并上传**
直接提交并推送。

Bash
    git add .
    git commit -m "feat: 快速实验成功"
    git push origin main
    ```

**情况 B：结果差，拉取回实验前的状态**
    直接清空所有未提交的修改，恢复到最后一次 commit 的状态。
    
```bash
    # 丢弃所有已跟踪文件的修改
    git restore .  
    # （或者使用老版本 Git 的命令：git reset --hard HEAD）

    # 删除所有未跟踪的新文件（如实验中产生的临时文件、新脚本等）
    # -f 表示强制，-d 表示包含目录
    git clean -fd
    ```


### 总结建议
*   **如果是长达几天、需要反复修改的实验：** 绝对使用 **方法一（分支）**。
*   **如果是只需几分钟、改几个参数的快速试错：** 可以使用 **方法二（重置）**。

## 结果
20000 checkpoint
243.78895828877688


### Experiment 2:
- `image_size`: 128
- `base_channels`: 128
- `batch_size`: 32
- `lr`: 1e-4
- `optimizer`: AdamW

- `epochs`: 1000
- `train_steps`: 100000
- `checkpoint_every_steps`: 5000
- `grad_clip`: 1.0

- `augment`: none
- `1crop_foreground`: true
- `pretrain`: false
- `sampling_steps`: 1000
- `use_EMA`: true