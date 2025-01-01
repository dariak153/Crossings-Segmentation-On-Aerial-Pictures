from dataclasses import dataclass
import torch

@dataclass
class DataConfig:
    images_dir: str = 'data/data'
    masks_dir: str = 'data/annotated_data/all_in_one'
    batch_size: int = 8
    num_workers: int = 2
    val_split: float = 0.2
    test_split: float = 0.1

@dataclass
class ModelConfig:
    num_classes: int = 3
    learning_rate: float = 1e-4
    pretrained: bool = True

@dataclass
class TrainerConfig:
    max_epochs: int = 50
    accelerator: str = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices: int = 1
    precision: int = 16 if torch.cuda.is_available() else 32
    log_every_n_steps: int = 10
    early_stopping_patience: int = 10
