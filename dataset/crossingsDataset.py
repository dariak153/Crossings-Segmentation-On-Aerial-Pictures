import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import albumentations as A
COLOR_MAP = {
    (0, 0, 0): 0, #backgroudn
    (255, 0, 0): 1,  # Class 1
    (0, 0, 255): 2  # Class 2
}


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform : A.Compose):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

        # Filter out images without corresponding masks
        self.image_mask_pairs = [
            (img, img.replace('.jpg', '.png')) for img in self.images
            if os.path.exists(os.path.join(masks_dir, img.replace('.jpg', '.png')))
        ]

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_name, mask_name = self.image_mask_pairs[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = np.asarray(Image.open(img_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path).convert("RGB"))

        image_copy = image.copy()
        mask_copy = mask.copy()

        #plt.imshow(mask)
        #plt.imshow(image)
        #plt.show()

        # Convert mask to class indices

        mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int8)
        for color, class_idx in COLOR_MAP.items():
            mask_class[np.all(mask == color, axis=-1)] = class_idx

        if self.transform:
            transformed = self.transform(image=image_copy,mask=mask_copy)
            image_copy = transformed["image"]
            mask_copy = transformed["mask"]

        # Convert to tensor

        mask_copy = torch.from_numpy(mask_class).long()
        image_copy = image_copy.permute(0, 1, 2).float()
        return image_copy, mask_copy

