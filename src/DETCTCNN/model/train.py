from argparse import ArgumentParser
from losses import DiceLoss, CEDiceLoss, FocalLoss
from model import get_model
import torch
from torch.utils.tensorboard import SummaryWriter


from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from  src.DETCTCNN.data import music_2d_dataset
from src.DETCTCNN.model.losses import DiceLossV2
from src.DETCTCNN.model.utils import calculate_min_max, class_weights, image_from_segmentation, plot_segmentation, calculate_data_statistics, standardize, normalize
MUSIC2DDataset = music_2d_dataset.MUSIC2DDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchio as tio
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
import numpy as np
import torch.backends

LABELS_SIZE = len(MUSIC_2D_LABELS)
device = torch.device("cuda" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))

def calculate_accuracy(pred_tensor, target_tensor):
    pred_tensor_flat = pred_tensor.cpu().argmax(dim=1).view(-1)
    target_tensor_flat = target_tensor.cpu().view(-1)

    correct = torch.eq(pred_tensor_flat, target_tensor_flat).sum().item()
    total_pixels = target_tensor_flat.numel()

    accuracy = (correct / total_pixels) * 100
    # print(f"accuracy: {accuracy}%")

    return accuracy

#TODO: Update training with dataloader
def main(hparams):
    
    # Initialize Transformations
    transform = music_2d_dataset.ImgAugTransform()

    
    transform = None
    train_dataset = MUSIC2DDataset(path2d=hparams.data_root, path3d=None,partition="train",spectrum="reducedSpectrum", transform=transform)
    if hparams.normalize_data:
        mean, std = calculate_data_statistics(train_dataset.images)
        train_dataset.images = list(map(lambda x: standardize(x,mean,std) , train_dataset.images))
        min, max  = calculate_min_max(train_dataset.images)
        train_dataset.images = list(map(lambda x: normalize(x,min,max) , train_dataset.images))
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)

    val_dataset = MUSIC2DDataset(path2d=hparams.data_root, path3d=None, partition="valid", spectrum="reducedSpectrum", transform=transform)
    if hparams.normalize_data:
        val_dataset.images = list(map(lambda x: standardize(x,mean,std) , val_dataset.images))
        val_dataset.images = list(map(lambda x: normalize(x,min,max) , val_dataset.images))
    val_loader = DataLoader(val_dataset)

    dice_weights = class_weights(dataset=train_dataset, n_classes=len(MUSIC_2D_LABELS))
    # Check dice weights used to weight loss function
    dice_weights = dice_weights.float().to(device=device)
    # print(dice_weights)


    model = get_model(input_channels=10, n_labels=hparams.n_labels, use_bn=True, basic_out_channel=2*64)
    model.to(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), betas=([0.9, 0.999]), lr = hparams.learning_rate)

    tb = SummaryWriter()


    # Metric: IOU
    jaccard = torchmetrics.JaccardIndex('multiclass', num_classes=LABELS_SIZE).to(device="cpu")
    loss_criterion = None
    if hparams.loss == "ce":
        # Use Weighted Cross Entropy
        loss_criterion = torch.nn.CrossEntropyLoss(weight=dice_weights).to(device)
    elif hparams.loss == "dice":
        # Use Weighted Dice Loss
        loss_criterion = DiceLossV2().to(device)
    elif hparams.loss == "focal":
        # Use Weighted Dice Loss
        loss_criterion = FocalLoss(gamma=2, alpha=dice_weights).to(device)
    else: # Use both losses
        loss_criterion = CEDiceLoss(weight=dice_weights, ce_weight=0.5).to(device)

    for epoch in range(hparams.epochs):  # loop over the dataset multiple times


        running_loss = 0.0
        train_accuracy = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y  = data['image'], data['segmentation'].argmax(1)
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            # Forward Pass
            y_hat = model(X)
            loss = 1000* loss_criterion(y_hat, y)

            # backward pass
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_accuracy = calculate_accuracy(pred_tensor=y_hat, target_tensor=y)
            train_iou = jaccard(y_hat.cpu().argmax(1), y.cpu()) * 100

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss
                }, "model.pt")

            tb.add_scalar("Loss", running_loss, epoch)
            tb.add_scalar("Train_acc", train_accuracy, epoch)
            tb.add_scalar("Train_IOU", train_iou, epoch)

            iteration = epoch * len(train_loader) + i
            if iteration % hparams.print_every == (hparams.print_every - 1):
                image_from_segmentation(y_hat, LABELS_SIZE, MUSIC_2D_PALETTE, device=device)
                print(f'[epoch: {epoch:03d}/iteration: {i :03d}] train_loss: {running_loss / hparams.print_every :.6f}, train_acc: {train_accuracy:.2f}%, train_IOU: {train_iou:.2f}%')
                running_loss = 0.
                train_accuracy = 0.
                train_iou = 0.

            # tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE, MUSIC_2D_PALETTE))
            #print('(Epoch: {} / {}) Train_Loss: {:.4f}, train_acc: {:.2f}%'.format(epoch + 1, hparams.epochs, running_loss / (1+(len(train_loader)*epoch+i)), train_accuracy / len(train_loader)))
            # if i % 10 == 9:
            #     tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE))
            #     print('(Epoch: {} / {}) Loss: {}'.format(epoch + 1, hparams.epochs, running_loss / (len(train_loader)*epoch+i)))

            if iteration % hparams.validate_every == (hparams.validate_every - 1):
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                val_iou = 0.0
                for val_data in val_loader:

                    val_X, val_y = val_data["image"].to(device), val_data["segmentation"].argmax(1).to(device)

                    with torch.no_grad():
                        val_pred = model(val_X)
                        loss = loss_criterion(val_pred, val_y)

                    val_loss +=loss.item()
                    val_acc += calculate_accuracy(val_pred, val_y)
                    val_iou += jaccard(y_hat.cpu().argmax(1), y.cpu()) * 100

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)
                val_iou /= len(val_loader)

                tb.add_scalar("Val_Loss", val_loss, epoch)
                tb.add_scalar("Val_Accuracy", val_acc, epoch)
                tb.add_scalar("Val_IOU", val_iou, epoch)
                print(f'[INFO-Validation][epoch: {epoch:03d}/iteration: {i :03d}] validation_loss: {val_loss:.6f}, validation_acc: {val_acc:.2f}%, validation_IOU: {val_iou:.2f}%')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    # parser.add_argument("-dr", "--data_root", type=str, default="/media/davidg-dl/Second SSD/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-ve", "--validate_every", type=int, default=10, help="Validate after each # of iterations")
    parser.add_argument("-pe", "--print_every", type=int, default=10, help="print info after each # of epochs")
    parser.add_argument("-e", "--epochs", type=int, default=3000, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.0001, help="Learning rate")
    parser.add_argument("-loss", "--loss", type=str, default="ce", help="Loss function")
    parser.add_argument("-n", "--normalize_data", type=bool, default=True, help="Loss function")
    args = parser.parse_args()
    main(args)