import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true, classes=3):
        logits = F.softmax(logits, dim=1)

        true_1_hot = F.one_hot(true, num_classes=classes).permute(0, 3, 1, 2).float()
        dims = (0,) + tuple(range(2, true.ndimension()))

        intersection = torch.sum(logits * true_1_hot, dims)
        cardinality = torch.sum(logits + true_1_hot, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth + 1e-7)
        loss = 1 - dice
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=None, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight_ce)
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice

    def forward(self, logits, true):

        # Cross Entropy Loss
        ce_loss = self.cross_entropy(logits, true)

        # Dice Loss
        dice_loss = self.dice_loss(logits, true)

        # Kombinacja strat
        return ce_loss + self.weight_dice * dice_loss
