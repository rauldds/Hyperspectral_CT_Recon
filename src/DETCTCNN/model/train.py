from argparse import ArgumentParser
from ast import arg
import datetime
from sys import path
from typing import Dict
from src.DETCTCNN.model.losses import CEDiceLoss, FocalLoss
from src.DETCTCNN.model.model import get_model
import torch
from torch.utils.tensorboard import SummaryWriter
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from  src.DETCTCNN.data import music_2d_dataset
from src.DETCTCNN.data.all_class_sampler import AllClassSampler
from src.DETCTCNN.model.losses import DiceLossV2
from src.DETCTCNN.model.metrics import mIoU_score
from src.DETCTCNN.model.utils import calculate_min_max, class_weights, class_weights_sklearn, image_from_segmentation, plot_segmentation, calculate_data_statistics, standardize, normalize
MUSIC2DDataset = music_2d_dataset.MUSIC2DDataset
from torch.utils.data import DataLoader
import torch.backends
import torchio as tio
import numpy as np
import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

# Number of classes in the dataset
LABELS_SIZE = len(MUSIC_2D_LABELS)
# RGB representation for each class
palette = np.array(MUSIC_2D_PALETTE)

# Define the device that will be used to train the neural net (cpu, cuda, mps)
device = torch.device("cuda" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))

def draw_inputs(scan):
    image_data = scan[9,:,:]
    plt.imshow(image_data)
    plt.axis('off')
    plt.title("Input")
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image

# Function to get the accuracy of the net predictions
def calculate_accuracy(pred_tensor, target_tensor):
    # Extract class values using argmax and then flatten the output tensor
    pred_tensor_flat = pred_tensor.cpu().argmax(dim=1).view(-1)
    target_tensor_flat = target_tensor.cpu().view(-1)

    # Check how many predicted class values of each the flattened target and prediction tensors are equal
    correct = torch.eq(pred_tensor_flat, target_tensor_flat).sum().item()
    total_pixels = target_tensor_flat.numel()

    accuracy = (correct / total_pixels) * 100
    # print(f"accuracy: {accuracy}%")

    return accuracy

def main(hparams):
    #REFERENCE LOSS TO SAVE THE MODEL
    ref_iou = 0
    print("Hyperparameters:")
    print(hparams)
    
    # Initialize Transformations
    transform = music_2d_dataset.JointTransform2D(crop=(hparams.patch_size, hparams.patch_size), p_flip=0.5, color_jitter_params=None, long_mask=True,erosion=hparams.erosion, p_random_affine=0.3)
    valid_transform = music_2d_dataset.JointTransform2D(crop=(96, 96), p_flip=0.5, color_jitter_params=None, long_mask=True, erosion=hparams.erosion)

    # Intialize Pad
    padding = (0, int(hparams.patch_size/2), int(hparams.patch_size/2))
    pad_transform = tio.Pad(padding)

    # Access the dataset folders
    path2d = hparams.data_root + "/MUSIC2D_HDF5"
    path3d = hparams.data_root + "/MUSIC3D_HDF5"

    # Number of used energy levels
    energy_levels = 10
    if hparams.spectrum != "reducedSpectrum":
        energy_levels = 128
    if hparams.dim_red != "none" or hparams.band_selection is not None:
        energy_levels = hparams.no_dim_red

    # Train dataset class definition
    print("Loading Training Data (And applying dim reduction)...")
    train_dataset = MUSIC2DDataset(
        path2d=path2d, path3d=path3d,
        partition="train", 
        spectrum=hparams.spectrum,
        transform=transform, 
        full_dataset=hparams.full_dataset, 
        dim_red = hparams.dim_red,
        no_dim_red = hparams.no_dim_red,
        band_selection = hparams.band_selection,
        include_nonthreat=True,
        oversample_2D=hparams.oversample_2D,
        split_file=hparams.split_file
    )

    # Extract the mean, standard deviation, min, and max from the dataset
    if hparams.normalize_data:
        mean, std = calculate_data_statistics(train_dataset.images)
        train_dataset.images = list(map(lambda x: standardize(x,mean,std) , train_dataset.images))
        min, max  = calculate_min_max(train_dataset.images)
        train_dataset.images = list(map(lambda x: normalize(x,min,max) , train_dataset.images))
    
    ################################################################
    #########       START: Configs for patch training     ##########
    ################################################################

    # DATASET FOR SAMPLER (NO TRANSFORM)
    ds_fs = MUSIC2DDataset(
        path2d=path2d, path3d=path3d,
        partition="train", 
        spectrum=hparams.spectrum,
        transform=None, 
        full_dataset=hparams.full_dataset, 
        dim_red = hparams.dim_red,
        no_dim_red = hparams.no_dim_red,
        band_selection = hparams.band_selection,
        include_nonthreat=True,
        oversample_2D=hparams.oversample_2D,
        split_file=hparams.split_file)
    our_sampler = AllClassSampler(data_source=ds_fs,batch_size=hparams.batch_size)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=hparams.batch_size,shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams.batch_size,sampler = our_sampler,drop_last=True)

    print("Loading Validation Data (And applying dim reduction)...")
    # Define the validation dataset class
    val_dataset = MUSIC2DDataset(
        path2d=path2d, path3d=path3d,
        partition="valid",
        spectrum=hparams.spectrum, 
        transform=valid_transform, 
        full_dataset=hparams.full_dataset,
        dim_red = hparams.dim_red,
        no_dim_red = hparams.no_dim_red,
        band_selection = hparams.band_selection,
        include_nonthreat=True,
        oversample_2D=1,
        split_file=hparams.split_file
    )

    # Extract the mean, standard deviation, min, and max from the validation dataset    
    if hparams.normalize_data:
        val_dataset.images = list(map(lambda x: standardize(x,mean,std) , val_dataset.images))
        val_dataset.images = list(map(lambda x: normalize(x,min,max) , val_dataset.images))

    val_loader = DataLoader(dataset=val_dataset, batch_size=64,shuffle=True)

    ################################################################
    #########       End: Configs for patch training     ##########
    ################################################################

    # Weights to define how representative is each class during loss estimation
    # weights_dataset = MUSIC2DDataset(
    #     path2d=path2d, path3d=path3d,
    #     partition="train", 
    #     spectrum=hparams.spectrum,
    #     transform=None, 
    #     full_dataset=hparams.full_dataset, 
    #     dim_red = hparams.dim_red,
    #     no_dim_red = hparams.no_dim_red,
    #     eliminate_empty=False,
    #     include_nonthreat=True,
    #     oversample_2D=1,
    #     split_file=hparams.split_file
    # )

    print("Generating Weights...")
    dice_weights = class_weights(dataset=train_dataset, n_classes=len(MUSIC_2D_LABELS))
    #dice_weights = class_weights_sklearn(dataset=weights_dataset, n_classes=len(MUSIC_2D_LABELS))

    # Check dice weights used to weight loss function
    dice_weights = dice_weights.float().to(device=device)
    print(dice_weights)

    # Call U-Net model
    print("Creating Model...")
    model = get_model(input_channels=energy_levels, n_labels=hparams.n_labels, use_bn=False, basic_out_channel=32, depth=hparams.network_depth, dropout=hparams.dropout)
    model.to(device=device)
    
    # Define ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas=([0.9, 0.999]), lr = hparams.learning_rate)

    # Define Learning Rate Scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.5)

    # Define Tensorboard writer
    tb = SummaryWriter(f'runs/{hparams.experiment_name}/patch_size_{hparams.patch_size}_{datetime.datetime.now().strftime("%b%d_%H-%M-%S")}')

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
        loss_criterion = FocalLoss(gamma=hparams.gamma, alpha=dice_weights, reduction=hparams.dice_reduc).to(device)
    else: # Use both losses
        loss_criterion = CEDiceLoss(weight=dice_weights, ce_weight=0.5).to(device)

    print("Training Started")
    for epoch in range(hparams.epochs):  # loop over the dataset multiple times
        model.train()
        # Initializing loss and accuracy
        running_loss = 0.0
        train_accuracy = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y  = data['image'], data['segmentation']
            if hparams.loss == "ce" or hparams.loss == "focal":
                y = y.argmax(1)
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            # Forward Pass
            y_hat = model(X)
            loss = loss_criterion(y_hat, y)
            
            if hparams.l1_reg == True:
                # Define L1 regularization strength (lambda)
                l1_lambda = 0.00000008  # Adjust this value as needed

                # Calculate L1 regularization term
                l1_reg = torch.tensor(0.,device=device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, p=1)
                # Add L1 regularization term to the loss
                loss += l1_lambda * l1_reg


            # backward pass
            loss.backward()
            optimizer.step()

            if hparams.loss == "dice" or hparams.loss == "cedice":
                y = y.cpu()
                y = y.argmax(1)

            # print statistics
            running_loss += loss.item()
            train_accuracy = calculate_accuracy(pred_tensor=y_hat, target_tensor=y)
            train_iou_per_class, train_iou = mIoU_score(y_hat.cpu().argmax(1), y.cpu(), n_classes=LABELS_SIZE)
            train_iou *= 100

        # Print training progress after X amount of epochs
        if epoch % hparams.print_every == (hparams.print_every - 1):
            # Write the training loss, accuracy, and IoU to tensorboard
            tb.add_scalar("Loss", running_loss, epoch)
            tb.add_scalar("Train_acc", train_accuracy, epoch)
            tb.add_scalar("Train_IOU", train_iou, epoch)
            # Write representative image of epoch to tensorboard
            # img_pred = y_hat[0]
            # img_input = X[0].detach().cpu().numpy()
            # tb_input_train = draw_inputs(img_input)
            # train_target = y[0].detach().cpu().numpy()
            # pred = img_pred.argmax(dim=0).detach().cpu().numpy()
            # colored_image = palette[pred]
            # colored_target = palette[train_target]
            # colored_image = torch.from_numpy(colored_image.astype(np.uint8))
            # target_image = torch.from_numpy(colored_target.astype(np.uint8))
            # tb.add_image("Pred Train Image", torch.transpose(colored_image, 0, 2), epoch)
            # tb.add_image("Input Train Image", tb_input_train, epoch)
            # tb.add_image("Target Train Image", torch.transpose(target_image, 0, 2), epoch)
            # image_from_segmentation(y_hat, LABELS_SIZE, MUSIC_2D_PALETTE, device=device)
            print(f'[epoch: {epoch:03d}/iteration: {i :03d}] train_loss: {running_loss / hparams.print_every :.6f}, train_acc: {train_accuracy:.2f}%, train_IOU: {train_iou:.2f}%')
            print(f'[epoch: {epoch:03d}/iteration: {i :03d}] train IOU per class in batch: {["{0:0.2f}".format(j) for j in train_iou_per_class]}')

            # tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE, MUSIC_2D_PALETTE))
            #print('(Epoch: {} / {}) Train_Loss: {:.4f}, train_acc: {:.2f}%'.format(epoch + 1, hparams.epochs, running_loss / (1+(len(train_loader)*epoch+i)), train_accuracy / len(train_loader)))
            # if i % 10 == 9:
            #     tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE))
            #     print('(Epoch: {} / {}) Loss: {}'.format(epoch + 1, hparams.epochs, running_loss / (len(train_loader)*epoch+i)))

        if epoch % hparams.validate_every == (hparams.validate_every - 1):
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_iou = 0.0
            val_iteration = 0
            val_iou_per_class = None
            # Get no of ocurrences per batch
            val_class_counts = torch.zeros(LABELS_SIZE)
            # Iterate over the whole validation dataset
            for val_data in val_loader:
                # Extract target and inputs
                val_X, val_y  = val_data['image'], val_data['segmentation']
                val_iteration += val_X.shape[0]
                if hparams.loss == "ce" or hparams.loss == "focal":
                    val_y = val_y.argmax(1)
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                with torch.no_grad():
                    # Generate prediction
                    val_pred = model(val_X)
                    loss = loss_criterion(val_pred, val_y)
                    if hparams.loss == "dice" or hparams.loss == "cedice":
                        val_y = val_y.cpu()
                        val_y = val_y.argmax(1)
                    #Convert prediction to an image (numpy array)
                    # img_pred = val_pred[0]
                    # img_val_target = val_y[0].detach().cpu().numpy()
                    # pred = img_pred.argmax(dim=0).detach().cpu().numpy()
                    # colored_image = palette[pred]
                    # colored_val_target = palette[img_val_target]
                    # colored_image = torch.from_numpy(colored_image.astype(np.uint8))
                    # val_image = torch.from_numpy(colored_val_target.astype(np.uint8))
                val_loss +=loss.item()
                val_acc += calculate_accuracy(val_pred, val_y)
                val_iou_per_class_cur, val_iou_cur = mIoU_score(val_pred.cpu().argmax(1), val_y.cpu(), n_classes=LABELS_SIZE)
                val_class_counts = val_class_counts + (torch.logical_not(torch.isnan(val_iou_per_class_cur))).long()
                val_iou += val_iou_cur * 100
                if val_iou_per_class == None:
                    val_iou_per_class = torch.nan_to_num(val_iou_per_class_cur)
                else:
                    val_iou_per_class += torch.nan_to_num(val_iou_per_class_cur)

                    
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_iou /= len(val_loader)
            val_iou_per_class /= val_class_counts
            val_loss = val_loss
            #Scheduler Step
            scheduler.step(val_iou)

            tb.add_scalar("Val_Loss", val_loss, epoch)
            tb.add_scalar("Val_Accuracy", val_acc, epoch)
            tb.add_scalar("Val_IOU", val_iou, epoch)
            #tb.add_image("Pred Val Image", torch.transpose(colored_image, 0, 2), epoch)
            #tb.add_image("Target Val Image", torch.transpose(val_image, 0, 2), epoch)
            print(f'[INFO-Validation][epoch: {epoch:03d}/iteration: {i :03d}] validation_loss: {val_loss:.6f}, validation_acc: {val_acc:.2f}%, validation_IOU: {val_iou:.2f}%')
            print(f'[INFO-Validation][epoch: {epoch:03d}/iteration: {i :03d}] validation IOU per class in batch: {["{0:0.2f}".format(j) for j in val_iou_per_class]}')

            # Save whenever the validation loss decreases
            if ref_iou < val_iou:

                ref_iou = val_iou
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss
                }, hparams.model_name + ".pt")
        if epoch == (hparams.epochs-1):
            tb.add_hparams(vars(hparams),
                           {"hparam/train_loss":running_loss, "hparam/train_accuracy":train_accuracy,
                           "hparam/train_IoU":train_iou, "hparam/valid_loss":val_loss,
                           "hparam/val_accuracy":val_acc, "hparam/val_iou":val_iou})

    torch.save({
        "epoch": hparams.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, hparams.model_name + "_final" + ".pt")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default=".", help="Data root directory")
    parser.add_argument("-ve", "--validate_every", type=int, default=20, help="Validate after each # of iterations")
    parser.add_argument("-pe", "--print_every", type=int, default=10, help="print info after each # of epochs")
    parser.add_argument("-e", "--epochs", type=int, default=4000, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0008, help="Learning rate")
    parser.add_argument("-loss", "--loss", type=str, default="focal", help="Loss function")
    parser.add_argument("-n", "--normalize_data", type=bool, default=False, help="Loss function")
    parser.add_argument("-sp", "--spectrum", type=str, default="fullSpectrum", help="Spectrum of MUSIC dataset")
    parser.add_argument("-ps", "--patch_size", type=int, default=40, help="2D patch size, should be multiple of 128")
    parser.add_argument("-dim_red", "--dim_red", choices=['none', 'pca', 'merge'], default="none", help="Use dimensionality reduction")
    parser.add_argument("-no_dim_red", "--no_dim_red", type=int, default=10, help="Target no. dimensions for dim reduction")
    parser.add_argument("-sample_strategy", "--sample_strategy", choices=['grid', 'label'], default="label", help="Type of sampler to use for patches")
    parser.add_argument("-fd", "--full_dataset", type=bool, default=True, help="Use 2D and 3D datasets or not")
    parser.add_argument("-dp", "--dropout", type=float, default=0.5, help="Dropout strenght")
    parser.add_argument("-nd", "--network_depth", type=float, default=3, help="Depth of Unet style network")
    parser.add_argument("-os2D", "--oversample_2D", type=int, default=1, help="Oversample 2D Samples")
    parser.add_argument("-dre", "--dice_reduc", type=str, default="mean", help="dice weights reduction method")
    parser.add_argument("-g", "--gamma", type=int, default=2, help="gamma of dice weights")
    parser.add_argument("-en", "--experiment_name", type=str, default="sampler", help="name of the experiment")
    parser.add_argument("-l1", "--l1_reg", type=bool, default=False, help="use l1 regularization?")
    parser.add_argument("-sf", "--split_file", type=bool, default=True, help="use pickle split")
    parser.add_argument("-bsel", "--band_selection", type=str, default=None, help="path to band list")
    parser.add_argument("-ls", "--label_smoothing", type=float, default=0.0, help="how much label smoothing")
    parser.add_argument("-ero", "--erosion", type=bool, default=True, help="apply erosion as augmention")
    parser.add_argument("-mn", "--model_name", type=str, default="model_new_10_09_2023_focal_bn", help="apply erosion as augmention")
    args = parser.parse_args()
    main(args)