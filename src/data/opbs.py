"""
Implementation of OPBS algorithm

Detail can be found in paper "A Geometry-Based Band Selection
Approach for Hyperspectral Image Analysis" in Algorithm 2
"""


import argparse
import os
import pickle
import numpy as np
import torch
# from MEV_SFS import load_gdal_data
# import scipy.io as scio

from src.DETCTCNN.data.music_2d_dataset import MUSIC2DDataset


def opbs(image_data, sel_band_count, removed_bands=None):
    if image_data is None:
        return None

    bands = image_data.shape[1]
    band_idx_map = np.arange(bands)

    if not (removed_bands is None):
        image_data = np.delete(image_data, removed_bands, 1)
        bands = bands - len(removed_bands)
        band_idx_map = np.delete(band_idx_map, removed_bands)

    # Compute covariance and variance for each band
    # TODO: data normalization to all band
    data_mean = np.mean(image_data, axis=0)
    image_data = image_data - data_mean
    data_var = np.var(image_data, axis=0)
    h = data_var * image_data.shape[0]
    op_y = image_data.transpose()

    sel_bands = np.array([np.argmax(data_var)])
    last_sel_band = sel_bands[0]
    current_selected_count = 1
    sum_info = h[last_sel_band]
    while current_selected_count < sel_band_count:
        for t in range(bands):
            if not (t in sel_bands):
                op_y[t] = op_y[t] - np.dot(op_y[last_sel_band], op_y[t]) / h[last_sel_band] * op_y[last_sel_band]

        max_h = 0
        new_sel_band = -1
        for t in range(bands):
            if not (t in sel_bands):
                h[t] = np.dot(op_y[t], op_y[t])
                if h[t] > max_h:
                    max_h = h[t]
                    new_sel_band = t
        sel_bands = np.append(sel_bands, new_sel_band)
        last_sel_band = new_sel_band
        sum_info += max_h
        estimate_percent = sum_info / (sum_info + (bands - sel_bands.shape[0]) * max_h)
        print(estimate_percent)
        current_selected_count += 1

    print(band_idx_map[sel_bands] + 1)
    print(np.sort(band_idx_map[sel_bands] + 1))

    return sel_bands

def save_bands(bands, file_name="selected_bands.pkl"):
    # store list in binary file so 'wb' mode
    with open(file_name, 'wb') as fp:
        pickle.dump(bands, fp)
        print('Done writing bands into a binary file')

def main(args):
    # remove_bands = [0, 1, 2, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    #                 111, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
    #                 157, 158, 159, 160, 161, 162, 163, 164, 216, 217, 218,
    #                 219]
    path2d = os.path.join(args.dataset, "MUSIC2D_HDF5")
    path3d = os.path.join(args.dataset, "MUSIC3D_HDF5")
    dataset = MUSIC2DDataset(
        path2d=path2d, 
        path3d=path3d,
        partition="train", 
        spectrum="fullSpectrum",
        transform=None, 
        full_dataset=True 
    )
    image_data = torch.stack([dataset[i]["image"].permute((1,2,0)) for i in range(len(dataset))]).view(-1,128).numpy()
    print(image_data.shape)
    y = torch.stack([dataset[i]["segmentation"].argmax(0) for i in range(len(dataset))]).view(-1).numpy()
    non_zero = np.nonzero(y > 0)
    image_data = image_data[non_zero]
    print("Applying OPBS band selection")
    bands = opbs(image_data, args.n_bands, None)
    save_bands(bands=bands)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", 
                           help="dataset path", type=str, 
                           default="./")
    argParser.add_argument("-f", "--file_name", 
                           help="Path to save selected bands", type=str, 
                           default="selected_bands_10.pkl")
    argParser.add_argument("-n", "--n_bands", 
                           help="Path to save selected bands", type=int, 
                           default=10)
    args = argParser.parse_args()
    main(args)
