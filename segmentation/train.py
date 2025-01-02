from segmentation.dataloaders.segmentation_dataloader import SegmentationDataModule
from segmentation.models.lightning_module import SegmentationLightningModule
from segmentation.models.unet import CustomSegmentationModel
from segmentation.config import TrainerConfig, DataConfig, ModelConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import lightning.pytorch as pl
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        images_dir=data_cfg.images_dir,
        masks_dir=data_cfg.masks_dir,
        transform=transform,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        val_split=data_cfg.val_split,
        test_split=data_cfg.test_split
    )
    datamodule.setup()


    if model_name == 'smp_unet':
        model = SegmentationLightningModule(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate,
            pretrained=model_cfg.pretrained
        )
    elif model_name == 'custom_unet':
        model = CustomSegmentationModel(
            num_classes=model_cfg.num_classes,
            lr=model_cfg.learning_rate
        )
    else:
        raise ValueError(f"Nieznany model: {model_name}")


    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=trainer_cfg.early_stopping_patience
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
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
