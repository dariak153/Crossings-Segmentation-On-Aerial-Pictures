import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes, weights=None, epsilon=1e-6):
        super(WeightedMulticlassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W)
        # targets: (N, H, W)

        if self.weights is None:
            self.weights = torch.ones(self.num_classes, device=inputs.device)

        inputs = F.softmax(inputs, dim=1)  # Apply softmax to get class probabilities

        total_loss = 0
        for c in range(self.num_classes):
            input_c = inputs[:, c]
            target_c = (targets == c).float()

            intersection = torch.sum(input_c * target_c)
            union = torch.sum(input_c) + torch.sum(target_c)

            dice_loss = 1 - (2. * intersection + self.epsilon) / (union + self.epsilon)
            weighted_dice_loss = self.weights[c] * dice_loss
            total_loss += weighted_dice_loss

        return total_loss / self.num_classes