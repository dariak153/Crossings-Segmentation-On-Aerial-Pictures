import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(images, masks, preds, indices, num_samples=10):
    NUM_CLASSES = 3

    color_map = {
        0: (0, 0, 0),        # Tło
        1: (255, 0, 0),      # Przejście dla pieszych
        2: (0, 0, 255),      # Przejazd dla rowerów

    }

    num_samples = min(num_samples, images.size(0))
    sample_indices = torch.randperm(images.size(0))[:num_samples]

    for i in sample_indices:
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        mask = masks[i].numpy()
        pred = preds[i].numpy()

        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        for cls, color in color_map.items():
            mask_rgb[mask == cls] = color
            pred_rgb[pred == cls] = color

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title(f"Image {indices[i].item()}")
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_rgb)
        plt.axis('off')

        plt.tight_layout()
        plt.show()