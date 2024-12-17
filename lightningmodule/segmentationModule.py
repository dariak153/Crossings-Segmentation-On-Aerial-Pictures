import torch.nn as nn
import torch
import lightning.pytorch as pl
import torch.optim as optim
from monai import losses
#import segmentation_models_pytorch as smp
from models import Unet
import matplotlib.pyplot as plt
from utils import diceScore
import torchmetrics
class MySegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        weights = torch.tensor([0.02, 0.49, 0.49])  # Example weights for the classes
        #self.loss_function = diceScore.WeightedMulticlassDiceLoss(num_classes=3, weights=weights)
        self.loss_function = losses.DiceCELoss(include_background=False,softmax=True,to_onehot_y=False)
        #self.loss_function = nn.CrossEntropyLoss()
        self.network = Unet.SegmentationModel()
        self.loss_function.requires_grad = True
        metrics = torchmetrics.MetricCollection(
            torchmetrics.Accuracy(task='multiclass', num_classes=3),
            #torchmetrics.Precision(task='multiclass',num_classes=3),
            #torchmetrics.Recall(task='multiclass',num_classes=3),s
            torchmetrics.F1Score(task='multiclass',num_classes=3)
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.loss_function(outputs,masks)
        self.train_metrics.update(outputs,masks)
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(self.train_metrics,  on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_function(outputs,masks)
        self.val_metrics.update(outputs, masks)
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Tym razem użyjmy optimizera Adam - uczenie powinno być szybsze
        return torch.optim.Adam(self.parameters(), lr=1e-4)