# UNet

Simple U-Net training project for binary pet segmentation on Oxford-IIIT Pet.

## Setup

```bash
uv sync
```

## Train

```bash
uv run python src/train.py
```

Example:

```bash
uv run python src/train.py --epochs 10 --batch-size 8 --num-workers 4
```

The dataset is downloaded automatically into `data/` on first run.

## TensorBoard

```bash
uv run tensorboard --logdir runs/unet
```

## Output

- Best checkpoint: `checkpoints/unet_best.pt`
- TensorBoard logs: `runs/unet`
