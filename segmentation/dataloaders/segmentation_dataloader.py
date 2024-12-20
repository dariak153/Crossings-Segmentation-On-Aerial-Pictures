from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
from ..datasets.segmentation_dataset import SegmentationDataset

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, images_dir, masks_dir, batch_size=16, num_workers=4, val_split=0.2, test_split=0.1):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        transform = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        full_dataset = SegmentationDataset(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            transform=transform
        )
        test_size = int(self.test_split * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
