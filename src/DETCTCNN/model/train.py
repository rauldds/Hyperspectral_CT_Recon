from argparse import ArgumentParser
from losses import WeightedLoss
from model import get_model
import torch
from ..data.music_2d_dataset import MUSIC2DDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO: Update training with dataloader
def main(hparams):
    model = get_model(n_labels=hparams.n_labels)
    dataset = MUSIC2DDataset(root=hparams.data_root,partition="train",spectrum="reducedSpectrum")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), betas=([0.9, 0.999]), lr = hparams.lr)
    criterion = WeightedLoss()
    # Sample data
    # x = torch.ones(size=(1,2,32,32,32))
    # y = model(x)
    for epoch in range(hparams.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y  = data['image'], data['segmentation'] 
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            y_hat = model(X).view(-1,15,2)

            loss = criterion(y, y_hat)

            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9: 
                print('(Epoch: {} / {}) Loss: {}'.format(epoch + 1, hparams.epochs, running_loss / (len(train_loader)*epoch+i)))



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../../MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--n_labels", type=int, default=11, help="Number of labels for final layer")
    parser.add_argument("--lr", type=int, default=0.00005, help="Learning rate")
    args = parser.parse_args()
    main(args)