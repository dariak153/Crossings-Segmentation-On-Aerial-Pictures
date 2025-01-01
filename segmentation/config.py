from dataclasses import dataclass
import torch
import os

@dataclass
class DataConfig:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir: str = os.path.join(parent_dir, 'data', 'data')
    masks_dir: str = os.path.join(parent_dir, 'data', 'annotated data', 'all_in_one')
    batch_size: int = 8
    num_workers: int = 2
    val_split: float = 0.2
    test_split: float = 0.1

    if not os.path.exists(images_dir):
        raise ValueError(f"Directory {images_dir} does not exist.")
    if not os.path.exists(masks_dir):
        raise ValueError(f"Directory {masks_dir} does not exist.")

@dataclass
class ModelConfig:
    num_classes: int = 3
    learning_rate: float = 1e-4
    pretrained: bool = True

@dataclass
class TrainerConfig:
    max_epochs: int = 70
    accelerator: str = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices: int = 1
    precision: int = 16 if torch.cuda.is_available() else 32
    log_every_n_steps: int = 10
