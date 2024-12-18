import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

COLOR_MAP = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 0, 255): 2
}

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.masks = [f for f in sorted(os.listdir(masks_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_mask_pairs = [(img, img) for img in self.images if img in self.masks]

        print(f"Znaleziono {len(self.image_mask_pairs)} sparowanych obrazów i masek.")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.image_mask_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = np.asarray(Image.open(img_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path).convert('RGB'))
        mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for color, class_idx in COLOR_MAP.items():
            mask_class[np.all(mask == color, axis=-1)] = class_idx

        if self.transform:
            transformed = self.transform(image=image, mask=mask_class)
            image = transformed['image']
            mask_class = transformed['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_class = torch.from_numpy(mask_class).long()

        return image, mask_class