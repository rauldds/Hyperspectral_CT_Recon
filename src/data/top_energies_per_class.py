from argparse import ArgumentParser
from matplotlib import pyplot as plt
import torch
from music_2d_dataset import MUSIC2DDataset
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from music_2d_labels import MUSIC_2D_LABELS
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA


PATH = os.path.join("experiments", "features")

def pca(data, k=20):
    pca = PCA(n_components=k)
    new_data = pca.fit_transform(data)
    return new_data

def feature_importance_per_material(args):
    """
    Calculates The importance of Hyperspectal Data using feature 
    importance tech for a binary task. We do a one vs all approach on
    our data
    """

    train_dataset = MUSIC2DDataset(path2d=args.data_root, path3d=None,partition="train",spectrum="fullSpectrum", transform=None)
    NO_FEATS = args.no_features

    # Stack all data, set per pixel
    X = torch.stack([train_dataset[i]["image"] for i in range(len(train_dataset))]).view(-1,128).numpy()
    y = torch.stack([train_dataset[i]["segmentation"].argmax(0) for i in range(len(train_dataset))]).view(-1,1).squeeze().numpy()

    if args.pca:
        X = pca(X)

    # Create Experiment directory
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
        if not os.path.exists(PATH):
            os.makedirs(PATH)

    good_feats = np.empty((1,),dtype=int)
    bad_feats = np.empty((1,), dtype=int)
    print(f"--------- Pixel-wise Binary Classifiers for {len(MUSIC_2D_LABELS)} Classes ---------")
    MODEL_PATH = None
    for label, val in tqdm(MUSIC_2D_LABELS.items()):
        y_cur = np.copy(y)

        # Convert into binary problem
        match_idx = (y_cur == val)
        y_cur[match_idx] = 1
        y_cur[~match_idx] = 0

        # Empty class: Skip
        if ~np.any(y_cur == 1):
            continue

        x_train, x_test, y_train, y_test = train_test_split(X, y_cur, test_size=0.20, random_state=0)

        # Create Model
        model = None
        if args.model == "linreg":
            model = LinearRegression()
        elif args.model == "logreg": 
            model = LogisticRegression()
        elif args.model == "dtree":
            model = DecisionTreeRegressor()

        MODEL_PATH = os.path.join(PATH, args.model)
        # fit the model
        model.fit(X=x_train, y=y_train)
        # get importance
        importance = None
        if args.model == "linreg":
            # Performs poorly
            importance = model.coef_
        elif args.model == "logreg": 
            # Performs well
            importance = model.coef_[0]
        elif args.model == "dtree":
            # Performs poorly
            importance = model.feature_importances_

        # Do permutation importance for class agnostic score
        if args.permutation:
            results = permutation_importance(model, X, y, n_repeats=5, random_state=0, n_jobs=args.permutation_cores)
            importance = results.importances_mean

        accuracy = model.score(x_test, y_test)
        worst = np.argpartition(importance, NO_FEATS)[:NO_FEATS]
        best = np.argpartition(importance, -NO_FEATS)[-NO_FEATS:]
        ids_importance = np.concatenate((worst, best))
        good_feats = np.concatenate((good_feats, best))
        bad_feats = np.concatenate((bad_feats, worst))

        # summarize feature importance
        if args.save:
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            f = open(f"{MODEL_PATH}/{label}.txt", "w")
            f.write(f"--------- Top {NO_FEATS} Worst and Best Features ----------\n")
            for i in ids_importance:
                f.write('Feature: {}, Score: {}\n'.format(i,importance[i]))
            f.write(f"Accuracy: {accuracy}")
            f.close()
        else:
            print(f"--------- Top {NO_FEATS} Worst and Best Features ----------")
            for i in ids_importance:
                print('Feature: {}, Score: {}'.format(i,importance[i]))
        # plot feature importance
        if args.save:
            plt.bar(np.char.mod('%d', ids_importance), importance[ids_importance])
            plt.title(f"Top and Bottom {NO_FEATS} Features for Class: {label.capitalize()}")
            plt.savefig(f"{MODEL_PATH}/{label}.png")
        plt.clf()
    good_feats = np.unique(good_feats[good_feats < 128], return_counts=True)
    filter_good = good_feats[1] > 1
    bad_feats = np.unique(bad_feats[bad_feats < 128], return_counts=True)
    filter_bad = bad_feats[1] > 1
    fig, ax = plt.subplots(2,1)
    fig.tight_layout(pad=2.0)
    # Plot the first bar chart
    ax[0].bar(np.char.mod('%d', good_feats[0][filter_good]), good_feats[1][filter_good], width=2)
    ax[0].set_title(f"Histogram top {NO_FEATS} features for all classes")

    # Plot the second bar chart on top of the first one
    ax[1].bar(np.char.mod('%d', bad_feats[0][filter_bad]), bad_feats[1][filter_bad], width=2)
    ax[1].set_title(f"Histogram bottom {NO_FEATS} features for all classes")
    fig.set_size_inches(14, 9)
    if args.save:
        plt.savefig(f"{MODEL_PATH}/histogram.png", dpi=600)
    else:
        plt.show()
    plt.clf()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-s", "--sample", type=int, default=0, help="Sample to Study")
    parser.add_argument("-sv", "--save", type=bool, default=True, help="Save Importances as Graphs")
    parser.add_argument("-n", "--no_features", type=int, default=10, help="Number of features to obtain from bottom and top")
    parser.add_argument("-model", "--model", choices=['linreg', 'logreg', 'dtree'], default="logreg", help="Model to use for importance")
    parser.add_argument("-p", "--permutation", type=bool, default=True, help="Use Permutation Importance Techniques")
    parser.add_argument("-p_cores", "--permutation_cores", type=int, default=-1, help="How many cores to use for permutations")
    parser.add_argument("-pca", "--pca", type=int, default=False, help="How many cores to use for permutations")
    args = parser.parse_args()
    feature_importance_per_material(args)
