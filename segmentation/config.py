from dataclasses import dataclass

@dataclass
class DataConfig:
    images_dir: str = "data/data"
    masks_dir: str = "data/annotated data/all_in_one"
    batch_size: int = 8
    num_workers: int = 4
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class ModelConfig:
    num_classes: int = 3
    learning_rate: float = 1e-4
    pretrained: bool = True

@dataclass
class TrainerConfig:
    max_epochs: int = 300
    accelerator: str = "gpu"
    devices: int = 1
    precision: int = 16
    log_every_n_steps: int = 5
    early_stopping_patience: int = 10
