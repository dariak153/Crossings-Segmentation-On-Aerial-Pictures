import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_predictions(images, masks, preds, indices, num_samples=5):
    samples = min(num_samples, images.size(0))
    for i in range(samples):
        idx = indices[i].item()
        img = images[i].detach().cpu().permute(1, 2, 0).numpy()
        true_mask = masks[i].detach().cpu().numpy()
        pred_mask = preds[i].detach().cpu().numpy()

        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img)
        axs[0].set_title(f"Image {idx}")
        axs[0].axis('off')
        axs[1].imshow(true_mask, cmap='jet', alpha=0.7)
        axs[1].set_title(f"Ground Truth {idx}")
        axs[1].axis('off')

        axs[2].imshow(pred_mask, cmap='jet', alpha=0.7)
        axs[2].set_title(f"Predicted Mask {idx}")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()
