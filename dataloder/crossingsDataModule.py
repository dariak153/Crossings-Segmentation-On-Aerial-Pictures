from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl

from dataset import crossingsDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, images_dir='../data/data', masks_dir='../data/annotated data/all_in_one', batch_size=4, num_workers=4, val_split=0.2, target_size=(512, 512)):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.target_size = target_size

        self.train_transform = A.Compose([
            A.Resize(*self.target_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.HueSaturationValue(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(*self.target_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        full_dataset = crossingsDataset.SegmentationDataset(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            transform=self.train_transform if stage == 'fit' else self.val_transform
        )
        train_size = int((1 - self.val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)