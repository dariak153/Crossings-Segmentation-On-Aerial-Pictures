import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from lightning.pytorch import LightningModule
from torchmetrics import Dice, JaccardIndex
from model2 import SegmentationModel


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

class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png'))])

        self.image_mask_pairs = [
            (img, mask) for img, mask in zip(self.images, self.masks) if img == mask
        ]

        assert len(self.image_mask_pairs) > 0, "Nie znaleziono pasujących obrazów i masek."

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
            augmented = self.transform(image=image, mask=mask_class)
            image = augmented['image']
            mask_class = augmented['mask'].long()

        return image, mask_class

def visualize_predictions(images, masks, predictions, num_samples=5):
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()

    for i in range(min(num_samples, len(images))):
        img = images[i].transpose(1, 2, 0)
        img = (img * 0.5) + 0.5  # Denormalizacja
        img = np.clip(img, 0, 1)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img)
        ax[0].set_title("Obraz")
        ax[0].axis('off')

        ax[1].imshow(masks[i], cmap='jet', vmin=0, vmax=NUM_CLASSES - 1)
        ax[1].set_title("Maska rzeczywista")
        ax[1].axis('off')

        ax[2].imshow(predictions[i], cmap='jet', vmin=0, vmax=NUM_CLASSES - 1)
        ax[2].set_title("Predykcja modelu")
        ax[2].axis('off')

        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Ewaluacja modelu segmentacji")
    parser.add_argument('--checkpoint', type=str, required=True, help='Ścieżka do checkpointu')
    parser.add_argument('--images_dir', type=str, required=True, help='Katalog z obrazami')
    parser.add_argument('--masks_dir', type=str, required=True, help='Katalog z maskami')
    parser.add_argument('--num_samples', type=int, default=5, help='Liczba próbek do wizualizacji')
    args = parser.parse_args()

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


    dataset = EvaluationDataset(args.images_dir, args.masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmentationModel.load_from_checkpoint(args.checkpoint, num_classes=NUM_CLASSES)
    model.to(device)
    model.eval()


    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            predictions = model(images)
            predictions = torch.argmax(predictions, dim=1)

        visualize_predictions(images, masks, predictions, num_samples=args.num_samples)

        if batch_idx >= args.num_samples - 1:
            break

if __name__ == "__main__":
    main()
