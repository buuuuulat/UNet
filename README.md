# UNet

Minimal U-Net project for binary pet segmentation on the Oxford-IIIT Pet dataset.

## Quickstart

```bash
uv sync
uv run python src/train.py --epochs 10 --batch-size 8 --num-workers 4
```

The dataset is downloaded automatically into `data/` on first run.

TensorBoard:

```bash
uv run tensorboard --logdir runs/unet
```

Run inference on selected test images:

```bash
uv run python src/test_model.py --checkpoint checkpoints/unet_best.pt --sample-idxs 0 5 12
```

Outputs:
- checkpoint: `checkpoints/unet_best.pt`
- TensorBoard logs: `runs/unet`
- prediction previews: `outputs/predictions/`

## Results

### Training Curves

<p align="center">
  <img src="other/images/loss_train.png" alt="Training loss" width="49%">
  <img src="other/images/loss_val.png" alt="Validation loss" width="49%">
</p>

<p align="center">
  <img src="other/images/dice_train.png" alt="Training Dice" width="49%">
  <img src="other/images/dice_val.png" alt="Validation Dice" width="49%">
</p>

<p align="center">
  <img src="other/images/dice_best_val.png" alt="Best validation Dice" width="72%">
</p>

### Examples

<p align="center">
  <img src="other/images/test_example_1.png" alt="Test example 1" width="100%">
</p>

<p align="center">
  <img src="other/images/test_example_2.png" alt="Test example 2" width="100%">
</p>
