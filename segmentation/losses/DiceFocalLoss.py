from monai.losses import DiceFocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class MonaiDiceFocalLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(MonaiDiceFocalLoss, self).__init__()
        self.loss = DiceFocalLoss(
            include_background=True,
            reduction='none',
            softmax=True,
        )
        # For 3 classes, e.g. [1.0, 1.0, 1.0]
        self.class_weights = torch.tensor(class_weights).cuda() if class_weights else None

    def forward(self, inputs, targets, per_image=False):
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        loss_map = self.loss(inputs, targets)  # [N, C, H, W]
        if self.class_weights is not None:
            # Multiply each class channel by its weight
            for c in range(loss_map.shape[1]):
                loss_map[:, c, :, :] *= self.class_weights[c]

        if per_image:
            # Mean over channel + spatial, one loss per image
            return loss_map.mean(dim=(1,2,3)).mean()  # if you want final scalar
        else:
            # Single global average
            return loss_map.mean()