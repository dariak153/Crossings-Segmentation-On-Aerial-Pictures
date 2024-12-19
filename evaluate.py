import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import cv2
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

        self.images = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in sorted(os.listdir(masks_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)


class PedestrianCyclingLightningModule(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=1e-4, pretrained=True):
        super().__init__()
        self.save_hyperparameters('num_classes', 'lr', 'pretrained')

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


def post_process_mask(pred_mask, kernel_size=3, min_area=80):
    post_mask = np.zeros_like(pred_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for cls in range(NUM_CLASSES):
        if cls == 0:
            continue
        cls_mask = (pred_mask == cls).astype(np.uint8)


        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                post_mask[labels == label] = cls

    return post_mask


def visualize_predictions(images, masks, preds, post_preds, num_samples=5):
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    for i in range(num_samples):
        img = images[i].transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        mask = masks[i]
        pred = preds[i]
        post_pred = post_preds[i]

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(img)
        ax[0].set_title("Obraz")
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='jet', vmin=0, vmax=NUM_CLASSES - 1)
        ax[1].set_title("Maski")
        ax[1].axis('off')

        ax[2].imshow(pred, cmap='jet', vmin=0, vmax=NUM_CLASSES - 1)
        ax[2].set_title("Predykcje z modelu")
        ax[2].axis('off')

        ax[3].imshow(post_pred, cmap='jet', vmin=0, vmax=NUM_CLASSES - 1)
        ax[3].set_title("Po post-processingu")
        ax[3].axis('off')

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja modelu")
    parser.add_argument('--checkpoint', type=str, required=True, help='Ścieżka do checkpointu')
    parser.add_argument('--images_dir', type=str, default='data/data', help='Katalog z obrazami')
    parser.add_argument('--masks_dir', type=str, default='data/annotated_data/all_in_one', help='Katalog z maskami')
    parser.add_argument('--num_samples', type=int, default=5, help='Liczba próbek do wizualizacji')
    args = parser.parse_args()

    datamodule = PedestrianCyclingDataModule(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        batch_size=16,
        num_workers=2,
        val_split=0.2,
        test_split=0.1
    )
    datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" {device}")

    try:
        model = PedestrianCyclingLightningModule.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            strict=False
        )
        model.to(device)
        model.eval()
        print(f"Model załadowany z {args.checkpoint}")
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return

    test_loader = datamodule.test_dataloader()

    try:
        images, masks = next(iter(test_loader))
    except Exception as e:
        print(f"Błąd podczas pobierania danych testowych: {e}")
        return

    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

    post_preds = []
    preds_np = preds.cpu().numpy()
    for pred in preds_np:
        post_pred = post_process_mask(pred, kernel_size=3, min_area=100)
        post_preds.append(post_pred)
    post_preds = np.array(post_preds)

    visualize_predictions(images, masks, preds, post_preds, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
