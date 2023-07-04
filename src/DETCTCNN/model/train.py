from argparse import ArgumentParser
from losses import WeightedLoss, DiceLoss, dice_loss
from model import get_model
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from src.DETCTCNN.augmentations.augmentations import AddGaussianNoise
from  src.DETCTCNN.data import music_2d_dataset
from src.DETCTCNN.model.utils import class_weights, image_from_segmentation, plot_segmentation
MUSIC2DDataset = music_2d_dataset.MUSIC2DDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchio as tio

LABELS_SIZE = len(MUSIC_2D_LABELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_accuracy(pred_tensor, target_tensor):
    pred_tensor_flat = pred_tensor.argmax(dim=1).view(-1)
    target_tensor_flat = target_tensor.argmax(dim=1).view(-1)

    correct = torch.eq(pred_tensor_flat, target_tensor_flat).sum().item()
    total_pixels = target_tensor_flat.numel()

    accuracy = (correct / total_pixels) * 100
    # print(f"accuracy: {accuracy}%")

    return accuracy


#TODO: Update training with dataloader
def main(hparams):
    
    # Initialize Transformations
    transform = tio.Compose([
        tio.RandomFlip(axes=(1,2)),
        tio.RandomNoise(std=(0,0.05)),
        # tio.RandomElasticDeformation(max_displacement=20),
    ])

    
    # transform = None
    train_dataset = MUSIC2DDataset(root=hparams.data_root,partition="train",spectrum="reducedSpectrum", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size)

    val_dataset = MUSIC2DDataset(root=hparams.data_root, partition="valid", spectrum="reducedSpectrum", transform=transform)
    val_loader = DataLoader(val_dataset)


    dice_weights = class_weights(dataset=train_dataset, n_classes=len(MUSIC_2D_LABELS))
    # Check dice weights used to weight loss function
    print(dice_weights)


    model = get_model(input_channels=10, n_labels=hparams.n_labels, use_bn=True)
    model
    
    optimizer = torch.optim.Adam(model.parameters(), betas=([0.9, 0.999]), lr = hparams.learning_rate)

    tb = SummaryWriter()

    for epoch in range(hparams.epochs):  # loop over the dataset multiple times

        loss_criterion = None
        if hparams.loss == "ce":
            # Use Weighted Cross Entropy
            loss_criterion = torch.nn.CrossEntropyLoss(weight=dice_weights).to(device)
        else:
            # Use Weighted Dice Loss
            loss_criterion = WeightedLoss(weights=dice_weights, loss_func=dice_loss).to(device)

        running_loss = 0.0
        train_accuracy = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y  = data['image'], data['segmentation']
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
            train_accuracy += calculate_accuracy(pred_tensor=y_hat, target_tensor=y)

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss
                }, "model.pt")

            tb.add_scalar("Loss", running_loss, epoch)
            tb.add_scalar("Train_acc", train_accuracy, epoch)

            iteration = epoch * len(train_loader) + i
            if iteration % hparams.print_every == (hparams.print_every - 1):
                image_from_segmentation(y_hat, LABELS_SIZE, MUSIC_2D_PALETTE)
                print(f'[epoch: {epoch:03d}/iteration: {i :03d}] train_loss: {running_loss / hparams.print_every :.6f}, train_acc: {train_accuracy:.2f}%')
                running_loss = 0.
                train_accuracy = 0.

            # tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE, MUSIC_2D_PALETTE))
            #print('(Epoch: {} / {}) Train_Loss: {:.4f}, train_acc: {:.2f}%'.format(epoch + 1, hparams.epochs, running_loss / (1+(len(train_loader)*epoch+i)), train_accuracy / len(train_loader)))
            # if i % 10 == 9:
            #     tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE))
            #     print('(Epoch: {} / {}) Loss: {}'.format(epoch + 1, hparams.epochs, running_loss / (len(train_loader)*epoch+i)))

            if iteration % hparams.validate_every == (hparams.validate_every - 1):
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                for val_data in val_loader:

                    val_X, val_y = val_data["image"].to(device), val_data["segmentation"].to(device)

                    with torch.no_grad():
                        val_pred = model(val_X)
                        loss = loss_criterion(val_pred, val_y)

                    val_loss +=loss.item()
                    val_acc += calculate_accuracy(val_pred, val_y)

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)

                tb.add_scalar("Val_Loss", val_loss, epoch)
                tb.add_scalar("Val_Accuracy", val_acc, epoch)
                print(f'[INFO-Validation][epoch: {epoch:03d}/iteration: {i :03d}] validation_loss: {val_loss:.6f}, validation_acc: {val_acc:.2f}%')


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-dr", "--data_root", type=str, default="/media/davidg-dl/Second SSD/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-ve", "--validate_every", type=int, default=10, help="Validate after each # of iterations")
    parser.add_argument("-pe", "--print_every", type=int, default=2, help="print info after each # of epochs")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.0005, help="Learning rate")
    parser.add_argument("-loss", "--loss", type=str, default="ce", help="Loss function")
    args = parser.parse_args()
    main(args)