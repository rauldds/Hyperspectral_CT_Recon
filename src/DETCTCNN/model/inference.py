from argparse import ArgumentParser
import numpy as np
from src.DETCTCNN.model.model import get_model
import torch
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS
from  src.DETCTCNN.data.music_2d_dataset import MUSIC2DDataset
LABELS_SIZE = len(MUSIC_2D_LABELS)

def main(hparams):
    dataset = MUSIC2DDataset(root=args.data_root,spectrum="reducedSpectrum",partition="valid")
    model = get_model(input_channels=10, n_labels=hparams.n_labels)
    model.type(torch.DoubleTensor)
    dataset = torch.from_numpy(np.asarray(dataset[0]["image"]))
    print(dataset.shape)
    with torch.no_grad():
        x = model(dataset)
    
    print(x)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-dr", "--data_root", type=str, default="../../../MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-e", "--epochs", type=int, default=700, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.00005, help="Learning rate")
    args = parser.parse_args()
    main(args)