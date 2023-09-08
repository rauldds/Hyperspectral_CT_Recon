from argparse import ArgumentParser
from sklearn import preprocessing, svm
from sklearn.discriminant_analysis import StandardScaler
import torch
from music_2d_dataset import MUSIC2DDataset
from sklearn.linear_model import LogisticRegression
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



PATH = os.path.join("experiments", "features")

def feature_importance_per_material(args):
    """
    Calculates The importance of Hyperspectal Data using feature 
    importance tech for a binary task. We do a one vs all approach on
    our data
    """

    train_dataset = MUSIC2DDataset(path2d=args.data_root, path3d=None,partition="train",spectrum="fullSpectrum", transform=None)
    test_samp = train_dataset[-2]["image"].permute(1,2,0).reshape(-1,128)
    test_samp_res = train_dataset[-1]["segmentation"].argmax(0)
    
    NO_FEATS = args.no_features


    # Stack all data, set per pixel
    sc = preprocessing.MinMaxScaler()
    X = torch.stack([train_dataset[i]["image"] for i in range(len(train_dataset))]).permute(0,2,3,1).reshape(-1,128).numpy()
    # X = sc.fit_transform(X)
    y = torch.stack([train_dataset[i]["segmentation"].argmax(0) for i in range(len(train_dataset))]).permute(1,2,0).reshape(-1,1).squeeze().numpy()
    # non_zero = np.nonzero(y > 0)
    # take_set = np.random.binomial(n=1, p=0.3, size=y.shape)
    # take_set = np.nonzero(take_set > 0)
    # non_zero = non_zero + take_set
    # X = X[non_zero]
    # y = y[non_zero]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=False)
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=5000,C=10, class_weight="balanced")
    # model =svm.SVC(kernel="poly", class_weight="balanced")
    print("Fitting model...")
    model.fit(X=x_train, y=y_train)
    
    accuracy = model.score(x_test, y_test)
    print(accuracy)

    y_pred = model.predict(X=test_samp)
    y_pred = y_pred.reshape(100,100)
    plt.imshow(y_pred, interpolation='nearest')
    plt.show()




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-s", "--sample", type=int, default=0, help="Sample to Study")
    parser.add_argument("-sv", "--save", type=bool, default=True, help="Save Importances as Graphs")
    parser.add_argument("-n", "--no_features", type=int, default=10, help="Number of features to obtain from bottom and top")
    parser.add_argument("-model", "--model", choices=['linreg', 'logreg', 'dtree'], default="logreg", help="Model to use for importance")
    parser.add_argument("-p", "--permutation", type=bool, default=True, help="Use Permutation Importance Techniques")
    parser.add_argument("-p_cores", "--permutation_cores", type=int, default=-1, help="How many cores to use for permutations")
    args = parser.parse_args()
    feature_importance_per_material(args)
