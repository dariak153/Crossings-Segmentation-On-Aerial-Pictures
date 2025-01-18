import torch
import os
import time
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from segmentation.config import DataConfig, ModelConfig, TrainerConfig
from segmentation.dataloaders.segmentation_dataloader import SegmentationDataModule
from segmentation.models.lightning_module import SegmentationLightningModule

def get_unique_run_dir(base_dir="checkpoints"):
    run_dir = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    full_path = os.path.join(base_dir, run_dir)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def export_to_torchscript(model, save_path="model_traced.pt"):
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 3, 512, 512)
    traced_model = torch.jit.trace(model.model, dummy_input)
    traced_model.save(save_path)
    print(f"Model zapisany w: {save_path}")

def train_model(model_name='smp_unet'):
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    trainer_cfg = TrainerConfig()

    transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.1),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    datamodule = SegmentationDataModule(
        config=data_cfg,
        transform=transform
    )
    datamodule.setup()

    if model_name == 'smp_unet':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='unet',
            backbone='resnet34'
        )
    elif model_name == 'deeplabv3plus_resnet34':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='deeplabv3plus',
            backbone='resnet34'
        )
    elif model_name == 'fpn_resnet34':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='fpn',
            backbone='resnet34'
        )
    elif model_name == 'unetplusplus_resnet34':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='unet++',
            backbone='resnet34'
        )
    elif model_name == 'unet_resnet18':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='unet',
            backbone='resnet18'
        )
    elif model_name == 'unet_mobilenetv2':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='unet',
            backbone='mobilenet_v2'
        )
    elif model_name == 'unet_effb0':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='unet',
            backbone='efficientnet-b0'
        )
    elif model_name == 'unetplusplus_mobilenetv2':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='unet++',
            backbone='mobilenet_v2'
        )
    elif model_name == 'deeplabv3_resnet34':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='deeplabv3plus',
            backbone='resnet34'
        )
    elif model_name == 'segformer_resnet50':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='segformer',
            backbone='resnet50'
        )
    elif model_name == "segformer_tu-semnasnet_100":
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='segformer',
            backbone='tu-semnasnet_100'
        )
    elif model_name == "segformer_mit_b0":
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained,
            model_type='segformer',
            backbone='mit_b0'
        )
    else:
        raise ValueError(f"Nieznany model: {model_name}")

    unique_dir = get_unique_run_dir(base_dir="checkpoints")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=unique_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=trainer_cfg.early_stopping_patience
    )
    logger = CSVLogger(save_dir="logs", name=f"{model_name}_logs")

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        precision='16-mixed' if trainer_cfg.precision == 16 else 32,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)

    ts_path = os.path.join(unique_dir, "model_traced.pt")
    export_to_torchscript(model, ts_path)

    print(f"Checkpoint zapisany w: {checkpoint_callback.best_model_path}")
    print(f"Model TorchScript zapisany w: {ts_path}")
