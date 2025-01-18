
import argparse

import torch.cuda
from segmentation.train import train_model

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
                            'segformer_resnet50',
                            'segformer_tu-semnasnet_100',
                            'segformer_mit_b0',
                        ],
                        default='smp_unet',
                        help='Wybierz model do trenowania')
    parser.add_argument('--epochs', type=int, default=None, help="Liczba epok")
    parser.add_argument('--batch_size', type=int, default=None, help="Rozmiar batcha")

    args = parser.parse_args()

    if args.epochs is not None:
        from segmentation.config import TrainerConfig
        TrainerConfig.max_epochs = args.epochs
    if args.batch_size is not None:
        from segmentation.config import DataConfig
        DataConfig.batch_size = args.batch_size
    torch.cuda.empty_cache()
    train_model(model_name=args.model)

if __name__ == "__main__":
    main()
