import torch

'''Returns per class and average IOU scores for classes in the batch'''
def mIoU_score(y_hat, y, smooth=1e-10, n_classes=16, ignore_index=-1):
    with torch.no_grad():
        y_hat = y_hat.contiguous().view(-1)
        y = y.contiguous().view(-1)

        iou_per_class = []
        for cls in range(0, n_classes): #loop per pixel class
            # ignore index from iou since we don't care
            true_class = y_hat == cls
            true_label = y == cls

            if true_label.sum().item() == 0: #no exist label in this loop
                iou_per_class.append(torch.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().item()
                union = torch.logical_or(true_class, true_label).sum().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        iou_per_class = torch.FloatTensor(iou_per_class)
        iou_per_class_mean = iou_per_class.clone().detach()
        if ignore_index >= 0 and ignore_index < n_classes:
            # Ignore class
            iou_per_class_mean[ignore_index] = torch.nan
        return [iou_per_class ,torch.nanmean(iou_per_class_mean)]