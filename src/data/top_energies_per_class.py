from argparse import ArgumentParser
from matplotlib import pyplot as plt

import torch
from music_2d_dataset import MUSIC2DDataset
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from music_2d_labels import MUSIC_2D_LABELS
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

PATH = os.path.join("experiments", "features")
##
def feature_importance_per_material(args):
    """
    Calculates The importance of Hyperspectal Data using feature 
    importance tech for a binary task. We do a one vs all approach on
    our data
    """

    train_dataset = MUSIC2DDataset(path2d=args.data_root, path3d=None,partition="train",spectrum="fullSpectrum", transform=None)

    # Stack all data, set per pixel
    X = torch.stack([train_dataset[i]["image"] for i in range(len(train_dataset))]).view(-1,128).numpy()
    y = torch.stack([train_dataset[i]["segmentation"].argmax(0) for i in range(len(train_dataset))]).view(-1,1).squeeze().numpy()

    # Create Experiment directory
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
        if not os.path.exists(PATH):
            os.makedirs(PATH)

    print(f"--------- Pixel-wise Binary Classifiers for {len(MUSIC_2D_LABELS)} Classes ---------")
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
            # Performs terribly
            importance = model.coef_
        elif args.model == "logreg": 
            # Performs well
            importance = model.coef_[0]
        elif args.model == "dtree":
            importance = model.feature_importances_

        accuracy = model.score(x_test, y_test)
        # worst = np.argpartition(importance, 5)[:5]
        # best = np.argpartition(importance, 5)[:5]
        # ids_importance = np.concatenate((worst, best))

        # summarize feature importance
        if args.save:
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            f = open(f"{MODEL_PATH}/{label}.txt", "w")
            for i,v in enumerate(importance):
                f.write('Feature: {}, Score: {}\n'.format(i,v))
            f.write(f"Accuracy: {accuracy}")
            f.close()
        else:
            for i,v in enumerate(importance):
                print('Feature: {}, Score: {}'.format(i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.title("Feature Importance class: " + label)
        if args.save:
            plt.savefig(f"{MODEL_PATH}/{label}.png")
        else:
            plt.show()
        plt.clf()

    # sample = train_dataset[args.sample]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-s", "--sample", type=int, default=0, help="Sample to Study")
    parser.add_argument("-sv", "--save", type=bool, default=True, help="Save Importances as Graphs")
    parser.add_argument("-model", "--model", choices=['linreg', 'logreg', 'dtree'], default="dtree", help="Model to use for importance")
    args = parser.parse_args()
    feature_importance_per_material(args)
