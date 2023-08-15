from argparse import ArgumentParser
from sys import path
from typing import Dict
from src.DETCTCNN.model.losses import DiceLoss, CEDiceLoss, FocalLoss
from src.DETCTCNN.model.model import get_model
import torch
from torch.utils.tensorboard import SummaryWriter
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from  src.DETCTCNN.data import music_2d_dataset
from src.DETCTCNN.model.losses import DiceLossV2
from src.DETCTCNN.model.metrics import mIoU_score
from src.DETCTCNN.model.utils import calculate_min_max, class_weights, image_from_segmentation, plot_segmentation, calculate_data_statistics, standardize, normalize
MUSIC2DDataset = music_2d_dataset.MUSIC2DDataset
from torch.utils.data import DataLoader
import torch.backends
import torchio as tio
import numpy as np


LABELS_SIZE = len(MUSIC_2D_LABELS)
palette = np.array(MUSIC_2D_PALETTE)

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
    transform = music_2d_dataset.JointTransform2D(crop=(hparams.patch_size, hparams.patch_size), p_flip=0.5, color_jitter_params=None, long_mask=True)
    
    # transform = None
    path2d = hparams.data_root + "/MUSIC2D_HDF5"
    path3d = hparams.data_root + "/MUSIC3D_HDF5"

    train_dataset = MUSIC2DDataset(
        path2d=path2d, path3d=path3d,
        partition="train", 
        spectrum=hparams.spectrum,
        transform=transform, 
        full_dataset=False, 
        dim_red = hparams.dim_red,
        no_dim_red = hparams.no_dim_red
    )
    
    if hparams.normalize_data:
        mean, std = calculate_data_statistics(train_dataset.images)
        train_dataset.images = list(map(lambda x: standardize(x,mean,std) , train_dataset.images))
        min, max  = calculate_min_max(train_dataset.images)
        train_dataset.images = list(map(lambda x: normalize(x,min,max) , train_dataset.images))
    
    # List to store samples from dataset classes
    train_list = []
    val_list = []

    # START CONFIGS to load train dataset for patch learning
    # Define how many patches to do per volume based on the patch size
    patches_for_full_volume = int(128/hparams.patch_size)*2

    #OVERRIDE PRINT_EVERY AND VALIDATE EVERY
    hparams.print_every = int(patches_for_full_volume*len(train_dataset)/hparams.batch_size)
    hparams.validate_every = hparams.print_every

    # Number of used energy levels
    energy_levels = 10
    if hparams.spectrum != "reducedSpectrum":
        energy_levels = 128
    if hparams.dim_red != "none":
        energy_levels = hparams.no_dim_red
    
    # # Convert elements from dataset class to torch io subjects and store them in a list
    # for data in (train_dataset):
    #     subject = tio.Subject(
    #         image=tio.ScalarImage(tensor=data["image"].unsqueeze(0)),
    #         segmentation=tio.LabelMap(
    #             tensor=torch.argmax(data["segmentation"],0).unsqueeze(0).repeat(1,energy_levels,1,1)
    #             ),
    #     )
    #     train_list.append(subject)
    
    # # Store all subjects in a torch io subjects dataset
    # TrainSubjectDataset = tio.data.SubjectsDataset(train_list)

    # # Define the type of sampler that'll be used to generate patches. GridSampler goes through the volume
    # # in an orderly manner. Other sampler alternatives sample randomly or based on a certain label.
    # # https://torchio.readthedocs.io/patches/patch_training.html
    # train_sampler = tio.GridSampler(patch_size=(energy_levels,
    #                                             hparams.patch_size,
    #                                             hparams.patch_size),
    #                                 subject=subject)

    # if hparams.sample_strategy == "label":
    #     label_probabilities : Dict[int, float] = {i: (0 if i == 0 else 1) for i in range(len(MUSIC_2D_LABELS))}
    #     train_sampler = tio.data.LabelSampler(
    #         patch_size=(energy_levels,
    #                     hparams.patch_size,
    #                     hparams.patch_size),
    #         label_probabilities=label_probabilities,
    #     )
    # # Queue that controls the loaded patches, provides them for each batch iteration
    # train_patches_queue = tio.Queue(
    #                                 TrainSubjectDataset,
    #                                 max_length=150,
    #                                 samples_per_volume=patches_for_full_volume,
    #                                 sampler=train_sampler,
    #                                 num_workers=1,
    # )
    # END CONFIGS
    
    # train_loader = DataLoader(train_patches_queue, batch_size=hparams.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams.batch_size)

    val_dataset = MUSIC2DDataset(
        path2d=path2d, path3d=path3d,
        partition="valid",
        spectrum=hparams.spectrum, 
        transform=transform, 
        full_dataset=False,
        dim_red = hparams.dim_red,
        no_dim_red = hparams.no_dim_red
    )
    
    if hparams.normalize_data:
        val_dataset.images = list(map(lambda x: standardize(x,mean,std) , val_dataset.images))
        val_dataset.images = list(map(lambda x: normalize(x,min,max) , val_dataset.images))
    
    # START CONFIGS to load validation dataset for patch learning (as described in lines above)
    # for data in (val_dataset):
    #     subject = tio.Subject(
    #         image=tio.ScalarImage(tensor=data["image"].unsqueeze(0)),
    #         segmentation=tio.LabelMap(
    #             tensor=torch.argmax(data["segmentation"],0).unsqueeze(0).repeat(1,energy_levels,1,1)
    #             ),
    #     )
    #     val_list.append(subject)
    # ValSubjectDataset = tio.data.SubjectsDataset(val_list)
    # val_sampler = tio.GridSampler(patch_size=(energy_levels,
    #                                             hparams.patch_size,
    #                                             hparams.patch_size),
    #                                 subject=subject)
    # val_patches_queue = tio.Queue(
    #                                 ValSubjectDataset,
    #                                 max_length=150,
    #                                 samples_per_volume=patches_for_full_volume,
    #                                 sampler=val_sampler,
    #                                 num_workers=1,
    # )
    # #END CONFIGS

    # val_loader = DataLoader(val_patches_queue)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hparams.batch_size)

    dice_weights = class_weights(dataset=train_dataset, n_classes=len(MUSIC_2D_LABELS))
    # Check dice weights used to weight loss function
    dice_weights = dice_weights.float().to(device=device)
    # print(dice_weights)


    model = get_model(input_channels=energy_levels, n_labels=hparams.n_labels, use_bn=True, basic_out_channel=16, depth=2, dropout=0.5)
    model.to(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), betas=([0.9, 0.999]), lr = hparams.learning_rate)

    tb = SummaryWriter()


    # Metric: IOU
    loss_criterion = None
    if hparams.loss == "ce":
        # Use Weighted Cross Entropy
        loss_criterion = torch.nn.CrossEntropyLoss(weight=dice_weights).to(device)
    elif hparams.loss == "dice":
        # Use Weighted Dice Loss
        loss_criterion = DiceLossV2().to(device)
    elif hparams.loss == "focal":
        # Use Weighted Dice Loss
        loss_criterion = FocalLoss(gamma=4, alpha=dice_weights).to(device)
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
            loss = loss_criterion(y_hat, y)

            # backward pass
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_accuracy = calculate_accuracy(pred_tensor=y_hat, target_tensor=y)
            train_iou = mIoU_score(y_hat.cpu().argmax(1), y.cpu(), n_classes=LABELS_SIZE) * 100

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

                    val_X, val_y  = val_data['image'], val_data['segmentation'].argmax(1)
                    val_X = val_X.to(device)
                    val_y = val_y.to(device)

                    with torch.no_grad():
                        val_pred = model(val_X)
                        loss = loss_criterion(val_pred, val_y)
                        pred = val_pred.argmax(dim=1).squeeze(0).detach().cpu().numpy()
                        colored_image = palette[pred]
                        colored_image = torch.from_numpy(colored_image.astype(np.uint8))


                    val_loss +=loss.item()
                    val_acc += calculate_accuracy(val_pred, val_y)
                    val_iou += mIoU_score(y_hat.cpu().argmax(1), y.cpu(), n_classes=LABELS_SIZE) * 100

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)
                val_iou /= len(val_loader)

                val_loss  = val_loss

                tb.add_scalar("Val_Loss", val_loss, epoch)
                tb.add_scalar("Val_Accuracy", val_acc, epoch)
                tb.add_scalar("Val_IOU", val_iou, epoch)
                tb.add_image("Val Image", torch.transpose(colored_image,0,2),epoch)
                print(f'[INFO-Validation][epoch: {epoch:03d}/iteration: {i :03d}] validation_loss: {val_loss:.6f}, validation_acc: {val_acc:.2f}%, validation_IOU: {val_iou:.2f}%')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon", help="Data root directory")
    parser.add_argument("-ve", "--validate_every", type=int, default=10, help="Validate after each # of iterations")

    parser.add_argument("-pe", "--print_every", type=int, default=10, help="print info after each # of epochs")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("-loss", "--loss", type=str, default="focal", help="Loss function")
    parser.add_argument("-n", "--normalize_data", type=bool, default=True, help="Loss function")
    parser.add_argument("-sp", "--spectrum", type=str, default="reducedSpectrum", help="Spectrum of MUSIC dataset")
    parser.add_argument("-ps", "--patch_size", type=int, default=80, help="2D patch size, should be multiple of 128")
    parser.add_argument("-dim_red", "--dim_red", choices=['none', 'pca'], default="none", help="Use dimensionality reduction")
    parser.add_argument("-no_dim_red", "--no_dim_red", type=int, default=5, help="Target no. dimensions for dim reduction")
    parser.add_argument("-sample_strategy", "--sample_strategy", choices=['grid', 'label'], default="label", help="Type of sampler to use for patches")
    args = parser.parse_args()
    main(args)