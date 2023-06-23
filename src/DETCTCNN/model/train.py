from argparse import ArgumentParser
from model import get_model
import torch

def main(hparams):
    model = get_model()
    x = torch.ones(size=(1,2,32,32,32))
    y = model(x)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../../MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("--max_epochs", type=int, default=200, help="Number of maximum training epochs")
    args = parser.parse_args()
    main(args)