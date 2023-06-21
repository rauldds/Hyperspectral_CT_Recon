import argparse
import os
import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

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

print(f'visualizing {classes[class_selection]} manual segmentation')
print(f'Located on: {DATASET_PATH+classes[class_selection]}/manualSegmentation/manualSegmentation.h5')

plt.imshow(data, interpolation='nearest')
plt.title("Manual Segmentation")
plt.show()

# Slices visualizations
with h5py.File(DATASET_PATH + "/" + classes[class_selection]
                + '/fullSpectrum/reconstruction/reconstruction.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F')
    data = data.squeeze(axis=1)
    print(data.shape)

fig, ax = plt.subplots()
plt.title("Slices")
plt.subplots_adjust(bottom=0.15)
ax.imshow(data[0])

# Update function
def update_slice(val):
    idx = int(sliderwave.val)
    ax.cla()
    ax.imshow(data[idx])
    fig.canvas.draw_idle()


# Sliders
axwave = plt.axes([0.25, 0.05, 0.5, 0.03])
sliderwave = Slider(axwave, 'Slice No.', 0, 127, valinit=0, valfmt='%d')
sliderwave.on_changed(update_slice)
plt.show()


# Sinograms visualizations
with h5py.File(DATASET_PATH + "/" + classes[class_selection]
                + '/fullSpectrum/projections/sinogram.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F')
    data = data.T

fig, ax = plt.subplots()
plt.title("Sinograms")
plt.subplots_adjust(bottom=0.15)
ax.imshow(data[0])

def update_sinogram(val):
    idx = int(sliderwave.val)
    ax.cla()
    ax.imshow(data[idx])
    fig.canvas.draw_idle()

axwave = plt.axes([0.25, 0.05, 0.5, 0.03])
sliderwave = Slider(axwave, 'Sinogram No.', 0, 127, valinit=0, valfmt='%d')
sliderwave.on_changed(update_sinogram)
plt.show()