import argparse
import os
import sys

import h5py
import numpy as np

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dataset", help="dataset path", type=str)

args = argParser.parse_args()
print("DATASET PATH: %s" % args.dataset)

# Path to the dataset
DATASET_PATH = args.dataset

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


# Manual Segmentation data visualization
with h5py.File(DATASET_PATH+ "/"+classes[class_selection]+'/manualSegmentation/manualSegmentation.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F').transpose()
    # print(data.shape)
    # data = data[127, :, :]
    # data = data.squeeze(axis=2)
    # data = data[:, :, 1]
#dataFile.close()

print(f'visualizing {classes[class_selection]} manual segmentation')
print(f'Located on: {DATASET_PATH+classes[class_selection]}/manualSegmentation/manualSegmentation.h5')

from matplotlib import pyplot as plt

plt.imshow(data, interpolation='nearest')
plt.show()

