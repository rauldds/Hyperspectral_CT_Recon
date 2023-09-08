from argparse import ArgumentParser
import torch
from music_2d_dataset import MUSIC2DDataset
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import music_2d_labels
from umap import UMAP


PATH = os.path.join("experiments", "features")


def pca_var(args):
    """
        This Script visualizes the explained variance with k components
        After applying PCA to our MUSIC2DDataset
    """

    # For PCA
    train_dataset = MUSIC2DDataset(path2d=args.data_root, path3d=None,partition="train",spectrum="fullSpectrum", transform=None)
    # Stack all data, set per pixel
    X = torch.stack([train_dataset[i]["image"].permute((1,2,0)) for i in range(len(train_dataset))]).view(-1,128).numpy()
    y = torch.stack([train_dataset[i]["segmentation"].argmax(0) for i in range(len(train_dataset))]).view(-1,1).numpy()

    # For umap
    umap_dataset = MUSIC2DDataset(path2d=args.data_root, path3d=None,partition="train",spectrum="reducedSpectrum", transform=None)
    # Stack all data, set per pixel
    X_umap = torch.stack([umap_dataset[i]["image"].permute((1,2,0)) for i in range(len(umap_dataset))]).view(-1,10).numpy()
    if not args.with_bg:
        nonzero = np.nonzero(y.squeeze() > 0)
        X = X[nonzero]
        X_umap = X_umap[nonzero]
        y = y[nonzero]


    # Results of PCA
    print("Fitting PCA data")
    pca = PCA(n_components=2, whiten=False)
    X = pca.fit_transform(X)

    # Results of umap
    print("Fitting UMAP data")
    umap = UMAP(n_components=2, init='random', random_state=0)
    X_umap = umap.fit_transform(X_umap)

    #Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    scatter = axs[0].scatter(X[:,0], X[:,1], c=y)
    axs[0].set_title('PCA Data')
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    axs[0].legend()
    # umap Plots
    axs[1].scatter(X_umap[:,0], X_umap[:,1], c=y)
    axs[1].set_title('Umap Data')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    axs[1].legend()
    plt.legend(handles=scatter.legend_elements()[0], labels=list(music_2d_labels.MUSIC_2D_LABELS.keys()))
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-s", "--sample", type=int, default=-1, help="Sample to Study")
    parser.add_argument("-sv", "--save", type=bool, default=True, help="Save Importances as Graphs")
    parser.add_argument("-wbg", "--with_bg", type=bool, default=True, help="Plot also background")
    args = parser.parse_args()
    pca_var(args)
