import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from torchmetrics import Dice, JaccardIndex

class CustomDeepLabV3Model(pl.LightningModule):
    def __init__(
        self,
        num_classes=3,
        lr=1e-4,
        pretrained=True,
        backbone="resnet34",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
            activation=None
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.dice_metric = Dice(num_classes=num_classes, average='macro')
        self.jaccard_metric = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)
        dice = self.dice_metric(preds, masks)
        iou = self.jaccard_metric(preds, masks)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)
        dice = self.dice_metric(preds, masks)
        iou = self.jaccard_metric(preds, masks)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_dice", dice, on_epoch=True, prog_bar=True)
        self.log("test_iou", iou, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
