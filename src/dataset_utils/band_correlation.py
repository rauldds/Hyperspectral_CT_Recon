import argparse
import os
import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
# importing required libraries of opencv
import cv2

def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dataset", help="dataset path", type=str)
argParser.add_argument("-b", "--band",default=61, help="dataset path", type=int)

args = argParser.parse_args()
print("DATASET PATH: %s" % args.dataset)
# Path to the dataset
DATASET_PATH = args.dataset

if str(DATASET_PATH) == "None":
    print("PLEASE PROVIDE THE DATASET PATH")
    exit()

# Print all classes to console
def get_classes(file_path=None):
    with open(file_path, 'r') as f:
        classes = f.read().splitlines()
        return classes
classes = get_classes(os.path.join(sys.path[0], '[MUSIC2D]class_names.txt'))

print(f'[INFO] please select the class to get a sample from: 0 - {len(classes) -1}')
for i, cls in enumerate(classes):
    print(f"{i} - {cls}")

class_selection = input("selection: ")
class_selection = int(class_selection)


# Slices visualizations
with h5py.File(DATASET_PATH + "/" + classes[class_selection]
                + '/fullSpectrum/reconstruction/reconstruction.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F')
    data = data.squeeze(axis=1)
    print(data.shape)

band = normalize8(data[args.band -1])
prev = normalize8(data[args.band -2])
next = normalize8(data[args.band])

histr = cv2.calcHist([band],[0],None,[10],[0,256])
histr2 = cv2.calcHist([prev],[0],None,[10],[0,256])
histr3 = cv2.calcHist([next],[0],None,[10],[0,256])
print(cv2.compareHist(histr,histr3,cv2.HISTCMP_CORREL))
plt.plot(histr2)
plt.show()
