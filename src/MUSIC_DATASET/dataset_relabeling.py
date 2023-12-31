"""
    This is a preprocessing script for the MUSIC_2D Dataset.
    The dataset is not appropriate for learning methods as is,
    So this script maps the segmentation maps to global pixel values
    to facilitate DL methods.
"""
import argparse
import h5py
import numpy as np
import os
from src.MUSIC_DATASET.utils.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_SAMPLES, MUSIC_3D_SAMPLES


argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dataset", help="dataset path", type=str, default="/Users/davidg/Documents/Hyperspectral_CT_Recon/MUSIC2D_HDF5")

args = argParser.parse_args()
print("DATASET PATH: %s" % args.dataset)
# Path to the dataset
DATASET_PATH = args.dataset

file_names=os.listdir(DATASET_PATH)
file_names.remove("README.md")
if "MUSIC2D" in DATASET_PATH:
    for folder in file_names:
        if "sample" in folder:
            with h5py.File(DATASET_PATH+ "/"+folder+'/manualSegmentation/manualSegmentation.h5', 'r') as f:
                data = np.array(f['data']['value'], order='F')
            data = data.tolist()
            for j in range (len(data)):
                for z in range(len(data[j])):
                    for i in MUSIC_2D_SAMPLES[folder]:
                        if data[j][z]==i:
                            data[j][z]=MUSIC_2D_SAMPLES[folder][i]
            data = np.asarray(data,int)
        else:
            with h5py.File(DATASET_PATH+ "/"+folder+'/manualSegmentation/manualSegmentation.h5', 'r') as f:
                data = np.array(f['data']['value'], order='F')
            data[data==2] = 13
            data[data==1] = MUSIC_2D_LABELS[folder]
        channels = []
        for clas in range(len(MUSIC_2D_LABELS)):
            channel = (data==clas)
            #print(channel.shape)
            channels.append(channel)
        channels = np.asarray(channels)
        hf = h5py.File(DATASET_PATH+ "/"+folder+'/manualSegmentation/manualSegmentation_global.h5', 'w')
        g1 = hf.create_group('data')
        g1.create_dataset("value",data=channels)
        hf.close()
elif "MUSIC3D" in DATASET_PATH:
    for folder in file_names:
        with h5py.File(DATASET_PATH+ "/"+folder+'/manualSegmentation/manualSegmentation.h5', 'r') as f:
            data = np.array(f['data']['value'], order='F')
        for (x, y, z), local_label in np.ndenumerate(data):
            data[x,y,z] = MUSIC_3D_SAMPLES[folder][local_label]
        channels = []
        for clas in range(len(MUSIC_2D_LABELS)):
            channel = (data==clas)
            #print(channel.shape)
            channels.append(channel)
        channels = np.transpose(np.asarray(channels),(1,0,2,3))
        hf = h5py.File(DATASET_PATH+ "/"+folder+'/manualSegmentation/manualSegmentation_global.h5', 'w')
        g1 = hf.create_group('data')
        g1.create_dataset("value",data=channels)
        hf.close()