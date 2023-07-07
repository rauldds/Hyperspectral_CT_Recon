import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
                dice_coef = dice_coef * self.weight  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss

class DiceLossV2(nn.Module):

    def __init__(self):
        super(DiceLossV2, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
    
class CEDiceLoss(nn.Module):
    def __init__(self, weight=None, ce_weight = 0.8):
        super(CEDiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight
        self.smooth = 1e-5
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLossV2()
        self.w = ce_weight

    def forward(self, predict, target):
         return (self.w) * self.ce(predict, target) + (1-self.w) * self.dice(predict, target)
    
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss