import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from .data_preprocess import test_ds
    from .models import UNet
    from .train import dice_score_from_logits, get_device, prepare_masks
except ImportError:
    from data_preprocess import test_ds
    from models import UNet
    from train import dice_score_from_logits, get_device, prepare_masks


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    return (image.cpu() * STD + MEAN).clamp(0.0, 1.0)


@torch.no_grad()
def save_prediction_preview(model, device, sample_idx: int, save_path: Path):
    if sample_idx < 0 or sample_idx >= len(test_ds):
        raise IndexError(f"Sample index {sample_idx} is out of range for test dataset of size {len(test_ds)}")

    image, mask = test_ds[sample_idx]

    image_batch = image.unsqueeze(0).to(device)
    target_batch = prepare_masks(mask.unsqueeze(0)).to(device)
    logits = model(image_batch)
    pred_mask = (torch.sigmoid(logits) > 0.5).float()
    sample_dice = dice_score_from_logits(logits, target_batch)

    image_to_plot = denormalize_image(image).permute(1, 2, 0).numpy()
    target_to_plot = target_batch.squeeze().cpu().numpy()
    pred_to_plot = pred_mask.squeeze().cpu().numpy()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_to_plot)
    axes[0].set_title("Image")
    axes[1].imshow(target_to_plot, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_to_plot, cmap="gray")
    axes[2].set_title(f"Prediction\nDice: {sample_dice:.4f}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return sample_dice


def build_save_path(output_dir: Path, sample_idx: int) -> Path:
    return output_dir / f"sample_{sample_idx}.png"


def main():
    parser = argparse.ArgumentParser(description="Run a trained U-Net on selected test images")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/unet_best.pt"),
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--sample-idxs",
        type=int,
        nargs="+",
        default=[0],
        help="One or more test sample indices to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory to save prediction preview images",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = get_device()
    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    dice_scores = []
    for sample_idx in args.sample_idxs:
        save_path = build_save_path(args.output_dir, sample_idx)
        sample_dice = save_prediction_preview(model, device, sample_idx, save_path)
        dice_scores.append(sample_dice)
        print(f"Sample {sample_idx}: dice={sample_dice:.4f} | saved to {save_path}")

    if len(dice_scores) > 1:
        mean_dice = sum(dice_scores) / len(dice_scores)
        print(f"Mean dice over selected samples: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
