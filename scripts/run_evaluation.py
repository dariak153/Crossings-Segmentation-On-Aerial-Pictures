
import sys
import os
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation.dataloaders.segmentation_dataloader import SegmentationDataModule
from segmentation.models.lightning_module import SegmentationLightningModule
from segmentation.models.unet import CustomSegmentationModel
from segmentation.visualization.visualize import visualize_predictions


def transform_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.encoder.layer'):
            # Zamiana 'model.encoder.layer1.0.conv1.weight' na 'model.encoder1.conv.0.weight'
            parts = k.split('.')
            if len(parts) >= 5:
                layer_num = parts[2]  # 'layer1'
                block_num = parts[3]  # '0'
                conv_part = parts[4]  # 'conv1.weight'
                conv_num = conv_part.replace('conv', '').split('.')[0]  # '1'
                rest = '.'.join(conv_part.split('.')[1:])  # 'weight'
                new_key = f"model.encoder{layer_num[-1]}.conv.{conv_num}.{rest}"
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def evaluate_model(model_type, checkpoint_path, images_dir, masks_dir, batch_size=8, num_workers=2, num_samples=5):
    # Inicjalizacja DataModule
    datamodule = SegmentationDataModule(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.0,
        test_split=1.0
    )
    datamodule.setup()

    # Ustalenie urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicjalizacja modelu na podstawie model_type
    if model_type == 'smp_unet':
        # Załaduj model SegmentationLightningModule z checkpointu
        model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
    elif model_type == 'custom_unet':
        # Inicjalizacja modelu
        model = CustomSegmentationModel(num_classes=3, lr=1e-4)

        # Załadowanie checkpointu
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Sprawdzenie, czy checkpoint zawiera 'state_dict'
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # Przekształcenie kluczy, jeśli jest to konieczne
        transformed_state_dict = transform_state_dict(state_dict)

        # Ładowanie state_dict z strict=False
        missing_keys, unexpected_keys = model.load_state_dict(transformed_state_dict, strict=False)
        if missing_keys:
            print(f"Brakujące klucze: {missing_keys}")
        if unexpected_keys:
            print(f"Niespodziewane klucze: {unexpected_keys}")
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

    # Przeniesienie modelu na urządzenie i ustawienie w trybie ewaluacji
    model.to(device)
    model.eval()
    print(f"Model '{model_type}' załadowany z {checkpoint_path}")

    # Pobranie loadera testowego
    test_loader = datamodule.test_dataloader()

    # Pobranie jednego batcha
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)

    # Ewaluacja modelu
    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

    # Wizualizacja wyników
    visualize_predictions(images, masks, preds, num_samples=num_samples)


def main():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir: str = os.path.join(parent_dir, 'data', 'data')
    masks_dir: str = os.path.join(parent_dir, 'data', 'annotated data', 'all_in_one')
    parser = argparse.ArgumentParser(description="Ewaluacja modelu segmentacji")
    parser.add_argument('--model', type=str, choices=['smp_unet', 'custom_unet'], required=True,
                        help='Typ modelu do ewaluacji: "smp_unet" lub "custom_unet"')
    parser.add_argument('--checkpoint', type=str, required=True, help='Ścieżka do checkpointu')
    parser.add_argument('--images_dir', type=str, default=images_dir, help='Katalog z obrazami')
    parser.add_argument('--masks_dir', type=str, default=masks_dir, help='Katalog z maskami')
    parser.add_argument('--batch_size', type=int, default=8, help='Rozmiar batcha')
    parser.add_argument('--num_workers', type=int, default=2, help='Liczba wątków do ładowania danych')
    parser.add_argument('--num_samples', type=int, default=5, help='Liczba próbek do wizualizacji')
    args = parser.parse_args()

    evaluate_model(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()

