from torchmetrics import Dice, JaccardIndex

def compute_metrics(preds, masks, num_classes=3):
    dice = Dice(num_classes=num_classes, average='macro')
    iou = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')
    return dice(preds, masks), iou(preds, masks)
