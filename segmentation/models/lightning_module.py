import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from torchmetrics import Dice, JaccardIndex
from segmentation.losses import CombinedLoss, DiceFocalLoss
from segmentation.config import ModelConfig

class SegmentationLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes=3,
        lr=1e-4,
        pretrained=True,
        backbone="resnet34",
        model_type='unet'
    ):
        super().__init__()
        self.save_hyperparameters()
        if model_type == 'unet':
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
                activation=None
            )
        elif model_type == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
                activation=None
            )
        elif model_type == 'fpn':
            self.model = smp.FPN(
                encoder_name=backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
                activation=None
            )
        elif model_type == 'unet++':
            self.model = smp.UnetPlusPlus(
                encoder_name=backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
                activation=None
            )
        #the best model
        elif model_type == 'segformer':
            self.model = smp.Segformer(
                encoder_name=backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
                activation=None
            )

        else:
            raise ValueError(f"Nieznany model_type: {model_type}")

        # Change the loss function here
        self.loss_fn = CombinedLoss(weight_ce=None, weight_dice=1.0, classes=num_classes)
        #self.loss_fn = DiceFocalLoss.MonaiDiceFocalLoss(class_weights=[0.004872, 0.204702, 0.790426])
        self.dice_metric = Dice(num_classes=num_classes, average='macro')
        self.jaccard_metric = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks, _ = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, _ = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)
        dice = self.dice_metric(preds, masks)
        iou = self.jaccard_metric(preds, masks)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, masks, idx = batch
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
