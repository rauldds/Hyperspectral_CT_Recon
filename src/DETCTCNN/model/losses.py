import torch
def dice_coefficient(y_true, y_pred, smooth=1., weight=None):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    if weight is None:
        intersection = torch.sum(y_true_f * y_pred_f)
    else:
        intersection = torch.sum(torch.flatten(weight) * y_true_f * y_pred_f)
    #return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return (2. * intersection + smooth) / (torch.sum(torch.square(y_true_f)) + torch.sum(torch.square(y_pred_f)) + smooth) # (torch.sum(torch.square(y_true), -1)


def dice_coefficient_loss(y_true, y_pred, n_labels):
    dice = 0
    y_pred = generate_seg(y_pred)
    for index in range(n_labels):
        dice += -dice_coefficient(y_true[:, :, :, :, index], y_pred[:, :, :, :, index])
    return dice


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return torch.mean(2. * (torch.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(torch.sum(y_true,
                                                            axis=axis) + torch.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    #y_pred = generate_seg(y_pred)
    # if isinstance(y_pred, list):
    #     y_pred = y_pred[0]
    return dice_coefficient(y_true[:, :, :, :, label_index], y_pred[:, :, :, :, label_index])



def generate_seg(prediction):  # tensor (N, D, H, W, C) , channel_last
    Max = torch.argmax(prediction)
    seg = torch.one_hot(Max, prediction.shape[-1])
    return seg


def weighted_loss(y_true, y_pred, weight, n_labels):
    # if isinstance(y_pred, list):
    #     y_pred = y_pred[0]
    loss = 0
    for index in range(n_labels):
        #loss += weighted_cross_entropy(y_true[:, :, :, :, index], y_pred[:, :, :, :, index], weight[:, :, :, :])
        loss += -1 * dice_coefficient(y_true[:, :, :, :, index], y_pred[:, :, :, :, index], weight=weight)#**1.01
    return loss


def weighted_cross_entropy(y_true, y_pred, weight):
    # CE = -sum( G * log(P))
    # weight D,H,W
    # add clip_by_value to avoid nan problem
    CE = torch.multiply(torch.flatten(y_true), torch.log(torch.clip_by_value(torch.flatten(y_pred), 1e-10, 1.0)))
    CE = torch.multiply(torch.flatten(weight), CE)
    return -torch.sum(CE) #-torch.reduce_sum(CE)

def get_median(v):
    v=torch.reshape(v,[-1])
    m = v.get_shape()[-1]//2  #channel last
    return torch.nn.top_k(v, m).values[m-1]
