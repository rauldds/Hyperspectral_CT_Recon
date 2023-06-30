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

LABELS_SIZE = len(MUSIC_2D_LABELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO: Update training with dataloader
def main(hparams):
    
    # Initialize Transformations
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=20),
        # transforms.RandomResizedCrop(),
    #     AddGaussianNoise(),
    #     transforms.ToTensor()
    # ])

    
    transform = None
    dataset = MUSIC2DDataset(root=hparams.data_root,partition="valid",spectrum="reducedSpectrum", transform=transform)
    train_loader = DataLoader(dataset, batch_size=hparams.batch_size)


    dice_weights = class_weights(dataset=dataset, n_classes=len(MUSIC_2D_LABELS))


    model = get_model(input_channels=10, n_labels=hparams.n_labels, use_bn=True)
    model.type(torch.DoubleTensor)
    
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
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y  = data['image'].type(torch.FloatTensor), data['segmentation']
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            # Forward Pass
            y_hat = model(X.type(torch.DoubleTensor))
            loss = loss_criterion(y_hat.type(torch.FloatTensor), y.type(torch.FloatTensor))
            print(torch.unique(y_hat.argmax(1), return_counts=True))
            print(torch.unique(y_hat.argmax(1), return_counts=True))

            # backward pass
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()


            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss
                }, "model.pt")

            # tb.add_scalar("Loss", running_loss, epoch)
            # tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE, MUSIC_2D_PALETTE))
            # print('(Epoch: {} / {}) Loss: {}'.format(epoch + 1, hparams.epochs, running_loss / (1+(len(train_loader)*epoch+i))))
            if i % 10 == 9: 
                tb.add_image(tag="Prediction" + str(i), global_step=len(train_loader)*epoch+i, img_tensor=image_from_segmentation(y_hat, LABELS_SIZE))
                print('(Epoch: {} / {}) Loss: {}'.format(epoch + 1, hparams.epochs, running_loss / (len(train_loader)*epoch+i)))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-e", "--epochs", type=int, default=700, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.0005, help="Learning rate")
    parser.add_argument("-loss", "--loss", type=str, default="ce", help="Loss function")
    args = parser.parse_args()
    main(args)