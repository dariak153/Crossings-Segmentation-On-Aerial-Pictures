
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation.train import train_model
from segmentation.config import TrainerConfig, DataConfig

import torch
torch.set_float32_matmul_precision('medium')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'smp_unet',
            'unet_resnet18',
            'unet_resnet34',
            'unet_mobilenetv2',
            'unet_effb0',
            'deeplabv3_resnet34'
        ],
        default='smp_unet',
        help="Modele do trenowania"
    )
    parser.add_argument('--epochs', type=int, default=None, help="Liczba epok")
    parser.add_argument('--batch_size', type=int, default=None, help="Rozmiar batcha")

    args = parser.parse_args()

    if args.epochs is not None:
        TrainerConfig.max_epochs = args.epochs
    if args.batch_size is not None:
        DataConfig.batch_size = args.batch_size

    train_model(model_name=args.model)

if __name__ == "__main__":
    main()

