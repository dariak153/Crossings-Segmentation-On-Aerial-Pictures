import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import Dice, JaccardIndex
from segmentation_models_pytorch import Unet
import lightning.pytorch as pl

COLOR_MAP = {
    (0, 0, 0): 0,      # Tło
    (255, 0, 0): 1,    # Przejście dla pieszych
    (0, 0, 255): 2     # Przejazd dla rowerów
}
NUM_CLASSES = len(COLOR_MAP)

def rgb_to_class(mask, color_map=COLOR_MAP):
    mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    for color, class_idx in color_map.items():
        mask_class[np.all(mask == color, axis=-1)] = class_idx
    return mask_class

class PedestrianCyclingSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png'))])
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png'))])

        self.image_mask_pairs = [
            (img, mask) for img, mask in zip(image_files, mask_files) if img == mask
        ]


    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, index):
        img_name, mask_name = self.image_mask_pairs[index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))
        mask_class = rgb_to_class(mask, COLOR_MAP)

        if self.transform:
            augmented = self.transform(image=image, mask=mask_class)
            image = augmented['image']
            mask_class = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_class = torch.from_numpy(mask_class).long()

        return image, mask_class

class SegmentationDataModule(LightningDataModule):
    def __init__(self, images_dir, masks_dir, transform, batch_size=8, val_split=0.2, test_split=0.1):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        dataset = PedestrianCyclingSegmentationDataset(self.images_dir, self.masks_dir, self.transform)

        total_size = len(dataset)
        test_size = int(self.test_split * total_size)
        val_size = int(self.val_split * total_size)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = EncodingBlock(3, features[0])
        self.encoder2 = EncodingBlock(features[0], features[1])
        self.encoder3 = EncodingBlock(features[1], features[2])
        self.encoder4 = EncodingBlock(features[2], features[3])

        self.bottleneck = EncodingBlock(features[3], features[3]*2)

        self.upconv4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.decoder4 = EncodingBlock(features[3]*2, features[3])

        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = EncodingBlock(features[2]*2, features[2])

        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = EncodingBlock(features[1]*2, features[1])

        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = EncodingBlock(features[0]*2, features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))


        bottleneck = self.bottleneck(self.pool(enc4))


        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)
        return out

class SegmentationModel(LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=1e-4):
        super().__init__()
        self.model = UNet(out_channels=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.dice_metric = Dice(num_classes=num_classes, average='macro')
        self.iou_metric = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)
        dice = self.dice_metric(preds, masks)
        iou = self.iou_metric(preds, masks)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

if __name__ == "__main__":
    base_dir = 'data'
    images_dir = os.path.join(base_dir, 'data')
    masks_dir = os.path.join(base_dir, 'annotated_data', 'all_in_one')

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    datamodule = SegmentationDataModule(
        images_dir, masks_dir, transform, batch_size=8
    )

    model = SegmentationModel(num_classes=NUM_CLASSES)

    logger = CSVLogger(save_dir="logs", name="segmentation_model")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='best-model', save_top_k=1, mode='min')

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10), checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, datamodule)

