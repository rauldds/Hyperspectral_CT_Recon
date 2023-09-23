'''
    This file performs a custom split of the MUSIC dataset.
    The split used in the experiments can be found in the 
    /splits folder
'''

from argparse import ArgumentParser
import pickle
import torch
from music_2d_labels import MUSIC_2D_LABELS
from  src.DETCTCNN.data import music_2d_dataset
MUSIC2DDataset = music_2d_dataset.MUSIC2DDataset
import numpy as np
LABELS_SIZE = len(MUSIC_2D_LABELS)
SAMPLES_2D=31


def get_distribution(segs):
    uniques, counts = np.unique(segs, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / (len(segs) * 100 * 100)))
    return uniques, percentages

def main(args):
    path2d = args.data_root + "/MUSIC2D_HDF5"
    path3d = args.data_root + "/MUSIC3D_HDF5"
    dataset = MUSIC2DDataset(
        path2d=path2d, path3d=path3d,
        partition="all", 
        spectrum="reducedSpectrum",
        transform=None, 
        full_dataset=True, 
        dim_red = "none",
        no_dim_red = 10,
        band_selection = None,
        include_nonthreat=True,
        oversample_2D=1,
        eliminate_empty=True
    )
  
    segs = torch.stack(dataset.segmentations).numpy()
    segs = segs.argmax(1)
    uniques, percentages = get_distribution(segs)

    train_idx = [i for i in range(0, SAMPLES_2D+1)]
    val_idx = [i for i in range(0, SAMPLES_2D+1)]
    for i in range(32,len(segs)):
        if i % 5 in [0,1,3,4]:
            train_idx.append(i)
        else:
            val_idx.append(i)
    print(f"Train has {len(train_idx)} samples")
    print(f"Val has {len(val_idx)} samples")
    val_data = segs[val_idx]
    train_data = segs[train_idx]
    u_train, per_train = get_distribution(train_data)
    u_val, per_val = get_distribution(val_data)
    print(u_train)
    print(per_train)
    print(u_val)
    print(per_val)
    split = {"train": train_idx, "valid": val_idx}
    with open('splits/four_one_split.pkl','wb') as f:
        pickle.dump(split, f)
    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon", help="Data root directory")
    parser.add_argument("-ve", "--validate_every", type=int, default=20, help="Validate after each # of iterations")
    parser.add_argument("-pe", "--print_every", type=int, default=10, help="print info after each # of epochs")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.00008, help="Learning rate")
    parser.add_argument("-loss", "--loss", type=str, default="ce", help="Loss function")
    parser.add_argument("-n", "--normalize_data", type=bool, default=False, help="Loss function")
    parser.add_argument("-sp", "--spectrum", type=str, default="reducedSpectrum", help="Spectrum of MUSIC dataset")
    parser.add_argument("-ps", "--patch_size", type=int, default=40, help="2D patch size, should be multiple of 128")
    parser.add_argument("-dim_red", "--dim_red", choices=['none', 'pca', 'merge'], default="none", help="Use dimensionality reduction")
    parser.add_argument("-no_dim_red", "--no_dim_red", type=int, default=10, help="Target no. dimensions for dim reduction")
    parser.add_argument("-sample_strategy", "--sample_strategy", choices=['grid', 'label'], default="label", help="Type of sampler to use for patches")
    parser.add_argument("-fd", "--full_dataset", type=bool, default=True, help="Use 2D and 3D datasets or not")
    parser.add_argument("-bsel", "--band_selection", type=str, default=None, help="path to band list")
    parser.add_argument("-ls", "--label_smoothing", type=float, default=0.0, help="how much label smoothing")
    parser.add_argument("-dp", "--dropout", type=float, default=0.5, help="Dropout strenght")
    parser.add_argument("-nd", "--network_depth", type=float, default=2, help="Depth of Unet style network")
    parser.add_argument("-os2D", "--oversample_2D", type=int, default=1, help="Oversample 2D Samples")
    parser.add_argument("-dre", "--dice_reduc", type=str, default="mean", help="dice weights reduction method")
    parser.add_argument("-g", "--gamma", type=int, default=4, help="gamma of dice weights")
    parser.add_argument("-en", "--experiment_name", type=str, default="reduced", help="name of the experiment")
    parser.add_argument("-l1", "--l1_reg", type=bool, default=True, help="use l1 regularization?")
    args = parser.parse_args()
    main(args)