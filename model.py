import os
import numpy as np
from PIL import Image
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Dice, JaccardIndex

import segmentation_models_pytorch as smp

from lightning.pytorch.loggers import CSVLogger

COLOR_MAP = {
    (0, 0, 0): 0,  # Tło
    (255, 0, 0): 1,  # Przejście dla pieszych
    (0, 0, 255): 2  # Przejazd dla rowerów
}
NUM_CLASSES = len(COLOR_MAP)

def rgb_to_class(mask, color_map):
    mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    for color, class_idx in color_map.items():
        mask_class[np.all(mask == color, axis=-1)] = class_idx
    return mask_class

class PedestrianCyclingSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.png'))]
        mask_files = [f for f in sorted(os.listdir(masks_dir)) if f.lower().endswith(('.png'))]

        self.image_mask_pairs = [
            (img, img) for img in self.images if img in mask_files
        ]

        print(f"Znaleziono {len(self.image_mask_pairs)} sparowanych obrazów i masek.")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.image_mask_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))
        mask_class = rgb_to_class(mask, COLOR_MAP)

        if self.transform:
            transformed = self.transform(image=image, mask=mask_class)
            image = transformed['image']
            mask_class = transformed['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_class = torch.from_numpy(mask_class).long()

        return image, mask_class

class PedestrianCyclingDataModule(pl.LightningDataModule):
    def __init__(self, images_dir, masks_dir, batch_size=8, num_workers=4, val_split=0.2, test_split=0.1):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

        self.transform = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        full_dataset = PedestrianCyclingSegmentationDataset(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            transform=self.transform
        )
        test_size = int(self.test_split * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

class PedestrianCyclingLightningModule(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=1e-4, pretrained=True):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
            activation=None
        )

        class_weights = torch.tensor([0.1, 0.45, 0.45], dtype=torch.float)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_metric = Dice(num_classes=num_classes, average='macro')
        self.jaccard_metric = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)
        dice = self.dice_metric(preds, masks)
        iou = self.jaccard_metric(preds, masks)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

if __name__ == "__main__":
    base_dir = 'data'
    images_dir = os.path.join(base_dir, 'data')
    masks_dir = os.path.join(base_dir, 'annotated_data', 'all_in_one')

    datamodule = PedestrianCyclingDataModule(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=16,
        num_workers=2,
        val_split=0.2,
        test_split=0.1
    )
    datamodule.setup()

    try:
        train_loader = datamodule.train_dataloader()
        images, masks = next(iter(train_loader))
        print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
    except Exception as e:
        print(f"Błąd podczas ładowania danych: {e}")
        exit(1)

    model = PedestrianCyclingLightningModule(num_classes=NUM_CLASSES, pretrained=True)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    csv_logger = CSVLogger("logs", name="segmentation_model")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stopping, checkpoint_callback],
        logger=csv_logger,
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32,
        accumulate_grad_batches=2,
    )

    trainer.fit(model, datamodule)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Najlepszy model zapisany w: {best_model_path}")