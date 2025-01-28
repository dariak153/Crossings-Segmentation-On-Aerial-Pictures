import torch
import os
import argparse
import sys

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from segmentation.visualization.visualize import visualize_predictions
from segmentation.dataloaders.segmentation_dataloader import SegmentationDataModule
from segmentation.config import DataConfig, ModelConfig, TrainerConfig
from segmentation.models.lightning_module import SegmentationLightningModule

def evaluate_ckpt(model_type, checkpoint_path, images, device, num_classes=3):
    if model_type.startswith('unetplusplus'):
        model_type_main = 'unet++'
        backbone = model_type.split('_')[-1]
        model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes,
            lr=1e-4,
            pretrained=True,
            backbone=backbone,
            model_type='unet++'
        )
    elif model_type.startswith('unet'):
        model_type_main = 'unet'
        backbone = model_type.split('_')[-1]
        model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes,
            lr=1e-4,
            pretrained=True,
            backbone=backbone,
            model_type='unet'
        )
    elif model_type.startswith('deeplabv3plus'):
        model_type_main = 'deeplabv3plus'
        backbone = model_type.split('_')[-1]
        model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes,
            lr=1e-4,
            pretrained=True,
            backbone=backbone,
            model_type='deeplabv3plus'
        )
    elif model_type.startswith('fpn'):
        model_type_main = 'fpn'
        backbone = model_type.split('_')[-1]
        model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes,
            lr=1e-4,
            pretrained=True,
            backbone=backbone,
            model_type='fpn'
        )
    elif model_type.startswith('segformer'):
        model_type_main = 'segformer'
        backbone = model_type.split('_')[-1]
        model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes,
            lr=1e-4,
            pretrained=True,
            backbone=backbone,
            model_type='segformer'
        )
    else:
        raise ValueError(f"Nieznany model_type z nazwy modelu: {model_type}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=[
                            'smp_unet',
                            'deeplabv3plus_resnet34',
                            'fpn_resnet34',
                            'unetplusplus_resnet34',
                            'unet_resnet18',
                            'unet_mobilenetv2',
                            'unet_effb0',
                            'unetplusplus_mobilenetv2',
                            'deeplabv3_resnet34',
                            'segformer_resnet50'
                        ],
                        required=True,
                        help='Wybierz model do ewaluacji: smp_unet, deeplabv3plus_resnet34, etc.'
                        )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Ścieżka do .ckpt')
    parser.add_argument('--pt_path', type=str, default=None,
                        help='Ścieżka do .pt')
    parser.add_argument('--format', type=str,
                        choices=['ckpt', 'torchscript'],
                        required=True,
                        help='Format modelu do ewaluacji (ckpt lub torchscript)')

    parser.add_argument('--images_dir', type=str, default=None, help='Katalog z obrazami')
    parser.add_argument('--masks_dir', type=str, default=None, help='Katalog z maskami')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=5)

    args = parser.parse_args()

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    trainer_cfg = TrainerConfig()

    if args.images_dir:
        data_cfg.images_dir = args.images_dir
    if args.masks_dir:
        data_cfg.masks_dir = args.masks_dir
    if args.batch_size:
        data_cfg.batch_size = args.batch_size
    if args.num_workers:
        data_cfg.num_workers = args.num_workers

    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    datamodule = SegmentationDataModule(
        config=data_cfg,
        transform=transform
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    try:
        batch = next(iter(test_loader))
    except StopIteration:
        raise ValueError("Dane niepoprawnie załadowane z zbioru testowego.")

    images, masks, indices = batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    masks = masks.to(device)

    if args.format == 'ckpt':
        if not args.checkpoint:
            raise ValueError("Musisz podać --checkpoint, gdy format=ckpt.")
        preds = evaluate_ckpt(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            images=images,
            device=device,
            num_classes=model_cfg.num_classes
        )
    elif args.format == 'torchscript':
        if not args.pt_path:
            raise ValueError("Musisz podać --pt_path, gdy format=torchscript.")
        preds = evaluate_torchscript(args.pt_path, images, device=device)
    else:
        raise ValueError("Nieznany format modelu.")

    preds = preds.cpu()
    images = images.cpu()
    masks = masks.cpu()

    visualize_predictions(images, masks, preds, indices, num_samples=args.num_samples)

if __name__ == "__main__":
    main()

