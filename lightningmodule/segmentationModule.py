import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.optim as optim
#import segmentation_models_pytorch as smp
from models import Unet
import matplotlib.pyplot as plt
from utils import diceScore
class MySegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        weights = torch.tensor([0.02, 0.49, 0.49])  # Example weights for the classes
        self.loss_function = diceScore.WeightedMulticlassDiceLoss(num_classes=3, weights=weights)
        self.network = Unet.SegmentationModel()
        self.loss_function.requires_grad = True
    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.loss_function(outputs,masks)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_function(outputs,masks)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=1e-3)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer