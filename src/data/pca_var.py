from argparse import ArgumentParser
import numpy
import torch
from music_2d_dataset import MUSIC2DDataset
import os
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt


PATH = os.path.join("experiments", "features")


def pca_var(args):
    """
        This Script visualizes the explained variance with k components
        After applying PCA to our MUSIC2DDataset
    """

    train_dataset = MUSIC2DDataset(path2d=args.data_root, path3d=None,partition="train",spectrum="fullSpectrum", transform=None)
    # Stack all data, set per pixel
    if args.sample != -1:
        X = train_dataset[args.sample]["image"].permute((1,2,0))
        X = X.reshape(-1,128).numpy()
    else:
        X = torch.stack([train_dataset[i]["image"].permute((1,2,0)) for i in range(len(train_dataset))]).view(-1,128).numpy()

    pca = None
    exp_var_pca = 0
    if args.method == "pca":
        pca = PCA(n_components=128, whiten=True)
        pca.fit(X)
        exp_var_pca = pca.explained_variance_ratio_
    elif args.method == "kpca":
        pca = KernelPCA(n_components=128, kernel="sigmoid")
        kpca_transform = pca.fit_transform(X)
        explained_variance = numpy.var(kpca_transform, axis=0)
        exp_var_pca = explained_variance / numpy.sum(explained_variance)
    cum_sum = exp_var_pca.cumsum()
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum)), cum_sum, where='mid',label='Cumulative explained variance')
    plt.axhline(y=0.9, color="red")
    plt.axvline(x=80, color="red")
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title(f"Explained Variance PCA")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-s", "--sample", type=int, default=20, help="Sample to Study")
    parser.add_argument("-sv", "--save", type=bool, default=True, help="Save Importances as Graphs")
    parser.add_argument("-method", "--method", choices=['pca', 'kpca'], default="pca", help="Method to use for dim red")
    args = parser.parse_args()
    pca_var(args)
