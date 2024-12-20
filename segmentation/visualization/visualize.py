import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(images, masks, preds, num_samples=5):
    for i in range(num_samples):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(images[i].cpu().permute(1, 2, 0))
        ax[0].set_title("Image")
        ax[1].imshow(masks[i].cpu(), cmap='jet')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(preds[i].cpu(), cmap='jet')
        ax[2].set_title("Prediction")
        plt.show()
