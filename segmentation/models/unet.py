import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torchmetrics import Dice, JaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class UNet(nn.Module):
    def __init__(self, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleDict()
        for idx, feature in enumerate(features, 1):
            if idx == 1:
                in_ch = 3
            else:
                in_ch = features[idx-2]
            self.encoder[f'conv{idx}'] = EncodingBlock(in_ch, feature)

        # Bottleneck
        self.bottleneck = EncodingBlock(features[-1], features[-1]*2)

        # Decoder
        self.upconv = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        reversed_features = features[::-1]
        for idx, feature in enumerate(reversed_features, 1):
            in_channels = features[-1]*2 // (2**(idx-1))
            self.upconv[f'up{idx}'] = nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2)
            self.decoder[f'conv{idx}'] = EncodingBlock(feature * 2, feature)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []
        out = x
        # Encoder
        for idx in range(1, 5):
            out = self.encoder[f'conv{idx}'](out)
            enc_features.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for idx in range(1, 5):
            out = self.upconv[f'up{idx}'](out)
            enc_feat = enc_features[-idx]
            out = torch.cat((out, enc_feat), dim=1)
            out = self.decoder[f'conv{idx}'](out)


        out = self.final_conv(out)
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
