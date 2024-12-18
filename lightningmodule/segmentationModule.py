import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.optim as optim
from monai import losses
#import segmentation_models_ptorch as smp
from monai.metrics import DiceMetric
from models import Unet
import matplotlib.pyplot as plt
from utils import diceScore
import torchmetrics
from models import Resnet

class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-5, input_channels=3):
        super().__init__()
        weights = torch.tensor([0.02, 0.49, 1.25])
        self.save_hyperparameters()
        self.model = Unet.SegmentationModel()
        #self.model = Resnet.SegmentationModel2(num_classes=num_classes, input_channels=input_channels)
        self.loss_fn = losses.DiceCELoss(to_onehot_y=True, softmax=True, include_background=True,weight=weights)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.dice_metric(logits, masks.unsqueeze(1))
        dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log("train_dice", dice, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.dice_metric(logits, masks.unsqueeze(1))
        dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.00005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }