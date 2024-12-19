import os
import numpy as np
from PIL import Image

COLOR_MAP = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 0, 255): 2,
}
NUM_CLASSES = len(COLOR_MAP)

def rgb_to_class(mask, color_map):
    mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    for color, class_idx in color_map.items():
        mask_class[np.all(mask == color, axis=-1)] = class_idx
    return mask_class

def get_file_pairs(images_dir, masks_dir):
    image_files = set(f for f in os.listdir(images_dir) if f.lower().endswith(('.png')))
    mask_files = set(f for f in os.listdir(masks_dir) if f.lower().endswith(('.png')))
    return [(os.path.join(images_dir, f), os.path.join(masks_dir, f)) for f in image_files & mask_files]

def calculate_class_pixel_counts(pairs, color_map):
    class_pixel_counts = np.zeros(len(color_map), dtype=np.int64)
    for _, mask_path in pairs:
        mask = np.array(Image.open(mask_path).convert('RGB'))
        mask_class = rgb_to_class(mask, color_map)
        for class_idx in range(len(color_map)):
            class_pixel_counts[class_idx] += np.sum(mask_class == class_idx)
    return class_pixel_counts

def calculate_class_weights(class_pixel_counts):
    total_pixels = np.sum(class_pixel_counts)
    class_weights = total_pixels / (class_pixel_counts + 1e-6)
    return class_weights / np.sum(class_weights)

def main(images_dir, masks_dir):
    pairs = get_file_pairs(images_dir, masks_dir)
    class_pixel_counts = calculate_class_pixel_counts(pairs, COLOR_MAP)
    class_weights = calculate_class_weights(class_pixel_counts)
    return class_weights, len(pairs)

if __name__ == "__main__":
    images_dir = os.path.join("data", "data")
    masks_dir = os.path.join("data", "annotated_data", "all_in_one")

    class_weights, pair_count = main(images_dir, masks_dir)

    print("Class Weights:")
    for idx, weight in enumerate(class_weights):
        print(f"Class {idx}: {weight:.6f}")
    print(f"Total pairs: {pair_count}")