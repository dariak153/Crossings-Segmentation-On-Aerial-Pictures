import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
from dataset import crossingsDataset
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, images_dir='../data/data', masks_dir='../data/annotated data/all_in_one', batch_size=1, num_workers=2, val_split=0.2):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        # Define transformations
        self.image_transform = A.Compose([
            A.Resize(512, 512),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.transforms.ToTensorV2(),
        ])
        # Load the dataset
        full_dataset = crossingsDataset.SegmentationDataset(images_dir=self.images_dir, masks_dir=self.masks_dir, transform=self.image_transform)

        # Define the train/validation split
        train_size = int((1 - self.val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

