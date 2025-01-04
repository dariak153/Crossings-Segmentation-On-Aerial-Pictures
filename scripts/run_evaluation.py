# scripts/run_evaluation.py

import torch
import os
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation.dataloaders.segmentation_dataloader import SegmentationDataModule
from segmentation.models.lightning_module import SegmentationLightningModule
from segmentation.models.unet import CustomSegmentationModel
from segmentation.models.deeplabv3 import CustomDeepLabV3Model
from segmentation.visualization.visualize import visualize_predictions

def evaluate_ckpt(model_type, checkpoint_path, images, device):
    if model_type in ['unet_resnet18', 'unet_mobilenetv2', 'unet_effb0', 'unet_resnet34']:
        backbone_dict = {
            'unet_resnet18':  'resnet18',
            'unet_mobilenetv2': 'mobilenet_v2',
            'unet_effb0': 'efficientnet-b0',
            'unet_resnet34': 'resnet34'
        }
        backbone = backbone_dict[model_type]
        model = CustomSegmentationModel.load_from_checkpoint(
            checkpoint_path,
            num_classes=3,
            lr=1e-4,
            pretrained=True,
            backbone=backbone,
            use_unetpp=True
        )
    elif model_type == 'deeplabv3_resnet34':
        model = CustomDeepLabV3Model.load_from_checkpoint(
            checkpoint_path,
            num_classes=3,
            lr=1e-4,
            pretrained=True,
            backbone="resnet34"
        )
    elif model_type == 'smp_unet':
        model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
    return preds

def evaluate_torchscript(ts_path, images, device):
    scripted_model = torch.jit.load(ts_path, map_location=device)
    scripted_model.eval()

    with torch.no_grad():
        preds = scripted_model(images)
        preds = torch.argmax(preds, dim=1)
    return preds

def main():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(parent_dir, 'data', 'data')
    masks_dir = os.path.join(parent_dir, 'data', 'annotated_data', 'all_in_one') #zmienić na annotated data

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=[
                            'smp_unet',
                            'unet_resnet18',
                            'unet_resnet34',
                            'unet_mobilenetv2',
                            'unet_effb0',
                            'deeplabv3_resnet34'
                        ],
                        required=True,
                        help='Wybierz model do ewaluacji: unet_resnet34, deeplabv3_resnet34, itp.'
                        )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Ścieżka do .ckpt (dla format=ckpt)')
    parser.add_argument('--pt_path', type=str, default=None,
                        help='Ścieżka do .pt (TorchScript)')
    parser.add_argument('--format', type=str,
                        choices=['ckpt', 'torchscript'],
                        required=True,
                        help='Format modelu do ewaluacji (ckpt lub torchscript)')

    parser.add_argument('--images_dir', type=str, default=images_dir, help='Katalog z obrazami')
    parser.add_argument('--masks_dir', type=str, default=masks_dir, help='Katalog z maskami')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5)

    args = parser.parse_args()
    datamodule = SegmentationDataModule(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.0,
        test_split=1.0
    )
    datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = datamodule.test_dataloader()
    images, masks, indices = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)
    if args.format == 'ckpt':
        if not args.checkpoint:
            raise ValueError("Musisz podać --checkpoint, gdy format=ckpt.")
        preds = evaluate_ckpt(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            images=images,
            device=device
        )
    elif args.format == 'torchscript':
        if not args.pt_path:
            raise ValueError("Musisz podać --pt_path, gdy format=torchscript.")
        preds = evaluate_torchscript(args.pt_path, images, device=device)
    else:
        raise ValueError("Nieznany format modelu (tylko ckpt lub torchscript).")

    visualize_predictions(images, masks, preds, indices, num_samples=args.num_samples)

if __name__ == "__main__":
    main()

