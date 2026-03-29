import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2


train_tf = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

test_tf = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

train_ds = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="segmentation",
    download=True,
)
train_ds = wrap_dataset_for_transforms_v2(train_ds)
train_ds.transforms = train_tf

test_ds = OxfordIIITPet(
    root="./data",
    split="test",
    target_types="segmentation",
    download=True,
)
test_ds = wrap_dataset_for_transforms_v2(test_ds)
test_ds.transforms = test_tf


def create_dataloaders(batch_size: int = 32, num_workers: int = 0):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
