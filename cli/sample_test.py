import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import your Lightning model class
from lightningmodule import segmentationModule

def load_model(checkpoint_path):
    model = segmentationModule.MySegmentationModel()
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


def visualize_prediction(image, mask, predicted_mask):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image.permute(1, 2, 0))  # Convert to HWC for display
    ax[0].set_title('Input Image')

    ax[1].imshow(mask)
    ax[1].set_title('Ground Truth Mask')

    ax[2].imshow(predicted_mask)
    ax[2].set_title('Predicted Mask')

    plt.show()


def main(image_path, mask_path, checkpoint_path):
    # Load the model
    model = load_model(checkpoint_path)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Ensure the image size matches your training size
        transforms.ToTensor()
    ])
    image = preprocess_image(image_path, transform)

    # Preprocess the mask (for visualization purposes)
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((512, 512))
    mask = np.array(mask)

    # Make prediction
    with torch.no_grad():
        predicted_mask = model(image).squeeze(0) # Remove batch dimension
        predicted_mask = torch.argmax(predicted_mask, dim=0).numpy() # Convert to numpy

        print(predicted_mask)
    # Visualize the prediction
    visualize_prediction(image.squeeze(0), mask, predicted_mask)

if __name__ == "__main__":
    image_path = "../data/data/071.png"
    mask_path = "../data/annotated data/all_in_one/071.png"  # If you have a corresponding mask for visualization
    checkpoint_path = "../saved_models/best_model.ckpt"
    num_classes = 3  # Change this to your number of classes
    weights = torch.tensor([0.2, 0.3, 0.5])  # Change this to your class weights

    main(image_path, mask_path, checkpoint_path)
