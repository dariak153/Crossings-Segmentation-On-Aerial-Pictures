import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

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

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        mask_files = sorted([
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])


        self.image_mask_pairs = [
            (img, img) for img in self.images if img in mask_files
        ]

        print(f"Znaleziono {len(self.image_mask_pairs)} par obrazów i masek.")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.image_mask_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)


        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))

        mask_class = rgb_to_class(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask_class)
            image = transformed['image']
            mask_class = transformed['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_class = torch.from_numpy(mask_class).long()

        return image, mask_class, idx
