import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torchmetrics import Dice, JaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = EncodingBlock(3, features[0])
        self.encoder2 = EncodingBlock(features[0], features[1])
        self.encoder3 = EncodingBlock(features[1], features[2])
        self.encoder4 = EncodingBlock(features[2], features[3])

        self.bottleneck = EncodingBlock(features[3], features[3]*2)

        self.upconv4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.decoder4 = EncodingBlock(features[3]*2, features[3])

        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = EncodingBlock(features[2]*2, features[2])

        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = EncodingBlock(features[1]*2, features[1])

        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = EncodingBlock(features[0]*2, features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)
        return out

class CustomSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-4):
        super(CustomSegmentationModel, self).__init__()
        self.model = UNet(out_channels=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.dice_metric = Dice(num_classes=num_classes, average='macro')
        self.iou_metric = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)
        dice = self.dice_metric(preds, masks)
        iou = self.iou_metric(preds, masks)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
