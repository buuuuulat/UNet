import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from .data_preprocess import create_dataloaders
    from .models import UNet
except ImportError:
    from data_preprocess import create_dataloaders
    from models import UNet


def prepare_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    Oxford-IIIT Pet segmentation masks contain classes:
    1 = pet, 2 = background, 3 = border.
    Convert them to a binary pet-vs-background target for BCE loss.
    """
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    return (masks == 1).float()


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    """
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W] float {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, desc: str = "Train"):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    progress = tqdm(loader, total=len(loader), desc=desc, leave=False) if tqdm is not None else loader

    for batch_idx, (images, masks) in enumerate(progress, start=1):
        images = images.to(device)
        masks = prepare_masks(masks).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score_from_logits(logits.detach(), masks)

        avg_loss = total_loss / batch_idx
        avg_dice = total_dice / batch_idx

        if tqdm is not None:
            progress.set_postfix(loss=f"{avg_loss:.4f}", dice=f"{avg_dice:.4f}")
        elif batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"{desc} | batch {batch_idx}/{len(loader)} | loss={avg_loss:.4f} dice={avg_dice:.4f}")

    return total_loss / len(loader), total_dice / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "Val"):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    progress = tqdm(loader, total=len(loader), desc=desc, leave=False) if tqdm is not None else loader

    for batch_idx, (images, masks) in enumerate(progress, start=1):
        images = images.to(device)
        masks = prepare_masks(masks).to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score_from_logits(logits, masks)

        avg_loss = total_loss / batch_idx
        avg_dice = total_dice / batch_idx

        if tqdm is not None:
            progress.set_postfix(loss=f"{avg_loss:.4f}", dice=f"{avg_dice:.4f}")
        elif batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"{desc} | batch {batch_idx}/{len(loader)} | loss={avg_loss:.4f} dice={avg_dice:.4f}")

    return total_loss / len(loader), total_dice / len(loader)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_summary_writer(log_dir: Path | None):
    if log_dir is None:
        return None
    if SummaryWriter is None:
        print("TensorBoard is not installed, skipping event logging.")
        return None
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard log dir: {log_dir}")
    return writer


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Oxford-IIIT Pet segmentation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for train and test loaders")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader worker processes")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/unet"), help="TensorBoard log directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/unet_best.pt"),
        help="Path to save the best model checkpoint",
    )
    args = parser.parse_args()

    device = get_device()
    train_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    writer = create_summary_writer(args.log_dir)

    best_val_dice = float("-inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size} | Num workers: {args.num_workers}")
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    if tqdm is None:
        print("tqdm is not installed, using periodic text progress instead.")
    if writer is not None:
        writer.add_text(
            "run/config",
            json.dumps(
                {
                    **vars(args),
                    "checkpoint": str(args.checkpoint),
                    "log_dir": str(args.log_dir),
                    "device": str(device),
                },
                indent=2,
            ),
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            desc=f"Train {epoch}/{args.epochs}",
        )
        val_loss, val_dice = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            desc=f"Val {epoch}/{args.epochs}",
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("dice/train", train_dice, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("dice/val", val_dice, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint to {args.checkpoint}")
            if writer is not None:
                writer.add_scalar("dice/best_val", best_val_dice, epoch)

    print(f"Best validation dice: {best_val_dice:.4f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
