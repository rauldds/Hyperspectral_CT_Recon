from argparse import ArgumentParser
from losses import WeightedLoss
from model import get_model
import torch

#TODO: Update training with dataloader
def main(hparams):
    model = get_model(n_labels=hparams.n_labels)
    optimizer = torch.optim.Adam(model.parameters(), betas=([0.9, 0.999]), lr = hparams.lr)
    criterion = WeightedLoss()
    x = torch.ones(size=(1,2,32,32,32))
    y = model(x)
    for epoch in range(hparams.epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y,y_hat)
        optimizer.step()

        # running_loss += loss.item()
        # if i % 10 == 9: 
        #     print('(Epoch: {} / {}) Loss: {}'.format(epoch + 1, epochs, running_loss / (len(trainloader)*epoch+i)))



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../../MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of maximum training epochs")
    parser.add_argument("--input_shape", type=int, default=(64,64,64), help="Number of maximum training epochs")
    parser.add_argument("--n_labels", type=int, default=7, help="Number of maximum training epochs")
    parser.add_argument("--lr", type=int, default=0.00005, help="Number of maximum training epochs")
    args = parser.parse_args()
    main(args)