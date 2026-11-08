import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self.dataset))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


def get_transforms():
    train_transform = A.Compose([
        A.Resize(400, 400),
        A.RandomCrop(380, 380),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(p=0.2),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.25, hue=0.07, p=0.4),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0), A.MotionBlur(blur_limit=5, p=1.0)], p=0.15),
        A.Perspective(scale=(0.02, 0.05), p=0.15),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform


def get_dataloaders(train_dir, val_dir, batch_size=32):
    train_transform, val_transform = get_transforms()

    train_dataset = CarDataset(train_dir, transform=train_transform)
    val_dataset = CarDataset(val_dir, transform=val_transform)

    class_counts = {}
    for _, label in train_dataset.dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    sample_weights = []
    for _, label in train_dataset.dataset.samples:
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print("Sınıf Dağılımı (Eğitim):")
    for cls_idx, cls_name in enumerate(train_dataset.classes):
        count = class_counts.get(cls_idx, 0)
        print(f"  {cls_name:20s}: {count:5d} görsel")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, train_dataset.classes
