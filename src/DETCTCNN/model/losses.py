import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    """Dice Loss PyTorch
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        C = predict.shape[1]
        intersection = torch.sum(predict * target, dim=(2,3))  # (N, C)
        union = torch.sum(predict.pow(2), dim=(2,3)) + torch.sum(target, dim=(2,3))  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss
    
class CEDiceLoss(nn.Module):
    def __init__(self, weight=None, ce_weight = 0.8):
        super(CEDiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight
        self.smooth = 1e-5
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(weight=weight)
        self.w = ce_weight

    def forward(self, predict, target):
         return (self.w) * self.ce(predict, target) + (1-self.w) * self.dice(predict, target)