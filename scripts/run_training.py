import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=['smp_unet', 'custom_unet'],
        default='smp_unet',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
    )

    args = parser.parse_args()


    if args.epochs is not None:
        from segmentation.config import TrainerConfig
        TrainerConfig.max_epochs = args.epochs

    if args.batch_size is not None:
        from segmentation.config import DataConfig
        DataConfig.batch_size = args.batch_size

    train_model(model_name=args.model)

if __name__ == "__main__":
    main()

