import torch
import torch.nn as nn

def dice_loss(input, target, smooth=1.):
    iflat = input.view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth)/((iflat * iflat).sum() + (tflat * tflat).sum() + smooth))

def weighted_loss(input, target, weights, loss_func, weighted_dimension=1):
    losses = torch.zeros(input.shape[weighted_dimension])
    for index in range(input.shape[weighted_dimension]):
        x = input.select(dim=weighted_dimension, index=index)
        y = target.select(dim=weighted_dimension, index=index)
        losses[index] = loss_func(x, y)
    return torch.mean(weights * losses)

class WeightedLoss(nn.Module):
    def __init__(self, weights, loss_func, weighted_dimension=1):
        super(WeightedLoss, self).__init__()
        self.weights = weights
        self.loss_func = loss_func
        self.weighted_dimension = weighted_dimension

    def forward(self, input, target):
        return weighted_loss(input=input, target=target, weights=self.weights, loss_func=self.loss_func,
                             weighted_dimension=self.weighted_dimension)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        return dice_loss(inputs, targets, smooth)