from segmentation.dataloaders.segmentation_dataloader import SegmentationDataModule
from segmentation.models.lightning_module import SegmentationLightningModule
from segmentation.config import TrainerConfig, DataConfig, ModelConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import lightning.pytorch as pl
import numpy as np
def train_model():
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    trainer_cfg = TrainerConfig()

    datamodule = SegmentationDataModule(
        images_dir=data_cfg.images_dir,
        masks_dir=data_cfg.masks_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        val_split=data_cfg.val_split,
        test_split=data_cfg.test_split
    )
    datamodule.setup()

    model = SegmentationLightningModule(
        num_classes=model_cfg.num_classes,
        lr=model_cfg.learning_rate,
        pretrained=model_cfg.pretrained
    )

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    csv_logger = CSVLogger("logs", name="segmentation_model")

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        callbacks=[early_stopping, checkpoint_callback],
        logger=csv_logger
    )

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_model()
