import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import random

# Import your Lightning model class
from lightningmodule import segmentationModule

def load_model(checkpoint_path):
    model = segmentationModule.SegmentationLightningModule()
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()
    return model

def preprocess_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    else:
        image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def visualize_predictions(images, masks, predicted_masks):
    fig, ax = plt.subplots(5, 3, figsize=(8, 8))  # Create a 5x3 grid

    for i in range(5):
        ax[i, 0].imshow(images[i].permute(1, 2, 0))  # Convert to HWC for display
        ax[i, 0].set_title(f'Input Image {i+1}')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(masks[i])
        ax[i, 1].set_title(f'Ground Truth Mask {i+1}')
        ax[i, 1].axis('off')

        ax[i, 2].imshow(predicted_masks[i])
        ax[i, 2].set_title(f'Predicted Mask {i+1}')
        ax[i, 2].axis('off')
    plt.tight_layout()
    plt.show()

def select_random_images(image_dir, num_images=5):
    all_images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    result = random.sample(all_images, num_images)
    print(result)
    return result

def main(image_dir, mask_dir, checkpoint_path):
    # Load the model
    model = load_model(checkpoint_path)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Ensure the image size matches your training size
        transforms.ToTensor()
    ])

    # Select 5 random images
    random_images = select_random_images(image_dir, 5)

    images = []
    masks = []
    predicted_masks = []

    for image_name in random_images:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)  # Assuming mask has the same name as image

        # Preprocess the image
        image = preprocess_image(image_path, transform)
        images.append(image.squeeze(0))  # Remove batch dimension for visualization

        # Preprocess the mask (for visualization purposes)
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((512, 512))
        mask = np.array(mask)
        masks.append(mask)

        # Make prediction
        with torch.no_grad():
            predicted_mask = model(image).squeeze(0)  # Remove batch dimension
            predicted_mask = torch.argmax(predicted_mask, dim=0).numpy()  # Convert to numpy
            predicted_masks.append(predicted_mask)

    # Visualize all predictions in one subplot
    visualize_predictions(images, masks, predicted_masks)

if __name__ == "__main__":
    image_dir = "../data/data/"
    mask_dir = "../data/annotated data/all_in_one/"  # Directory containing corresponding masks for visualization
    checkpoint_path = "../saved_models/best_model.ckpt"
    num_classes = 3  # Change this to your number of classes
    weights = torch.tensor([0.2, 0.3, 0.5])  # Change this to your class weights

    main(image_dir, mask_dir, checkpoint_path)
