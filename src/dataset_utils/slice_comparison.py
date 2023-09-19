import argparse

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt2
from matplotlib.widgets import Slider
import open3d as o3d
from collections import Counter
import os
import torch
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
palette = np.array(MUSIC_2D_PALETTE)
idx = 0
step = 0
idx_seg = 0

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)

def pointcloud_converter(data):
    '''Function to convert the segmented slices into a point cloud for 3D visualization'''
    points = []
    classes = []
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                if data[x,y,z]!=0:
                    point = [[x],[y],[z]]
                    points.append(point)
                    classes.append(data[x,y,z])
    points = (np.asarray(points)).squeeze(2)
    return points, classes

def pointcloud_colorizer(classes, num_classes):
    "Function to assign a random color to the different classes that appear in the cloud"
    colors = []
    point_colors = []
    '''for i in range (num_classes):
        color = list(np.random.choice(range(256), size=3))
        colors.append(color)
    colors =(np.asarray(colors))/255'''
    colors = [[0, 0, 1],
               [0, 1, 0],
               [1, 0, 0],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 0],
               [0, 0, 0]]
    colors = np.asarray(colors, dtype=float)
    print(colors.shape)

    for idx in classes:
        point_colors.append(colors[int(idx)-1])
    point_colors=np.asarray(point_colors)

    return point_colors

def update_energy_level(val):
    global idx, step
    idx = int(slider_energy.val)
    ax.cla()
    ax.imshow(data[idx][step,:,:])
    fig.canvas.draw_idle()

def update_step(val):
    global idx, step
    step = int(slider_step.val)
    ax.cla()
    ax.imshow(data[idx][step,:,:])
    fig.canvas.draw_idle()

def update_seg(val):
    global idx_seg
    step = int(slider_seg.val)
    ax.cla()
    ax.imshow(segmentation[step, :,:])
    fig.canvas.draw_idle()


argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dataset", help="dataset path", type=str, default="../../MUSIC3D_HDF5")

args = argParser.parse_args()
print("DATASET PATH: %s" % args.dataset)
# Path to the dataset
DATASET_PATH = args.dataset

if str(DATASET_PATH) == "None":
    print("PLEASE PROVIDE THE DATASET PATH")
    exit()

# BEGIN: SAMPLE SELECTION
file_names=os.listdir(DATASET_PATH)
file_names.remove("README.md")
count = -1
for f in file_names:
    count = count + 1
    print("[%s]" % count + f)

while True:
    ans_file = int(input("Select sample (number): "))
    if ans_file > count:
        print("Wrong selection.")
        continue
    file = file_names[ans_file]
    print("Selected sample: %s " % file)
    break

# END: SAMPLE SELECTION

with h5py.File(DATASET_PATH+ "/"+file+'/manualSegmentation/manualSegmentation_global.h5', 'r') as f:
    segmentation = np.array(f['data']['value'], order='F',dtype=np.int16)
#Visualization of Slices
with h5py.File(DATASET_PATH + "/" + file
                + '/reducedSpectrum/reconstruction/reconstruction.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F')

data = torch.from_numpy(data)
segmentation = torch.from_numpy(segmentation).argmax(1)
segmentation = np.asarray(palette[segmentation])

fig, ax = plt.subplots()
plt.title("Slices")
plt.subplots_adjust(bottom=0.15)
ax.imshow(data[0][0,:,:])

ax_energy = plt.axes([0.25, 0.05, 0.5, 0.03])
slider_energy = Slider(ax_energy, 'Sinogram No.', 0, data.shape[1]-1, valinit=0, valfmt='%d')
slider_energy.on_changed(update_energy_level)
ax_step = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_step = Slider(ax_step, 'Step No.', 0, data.shape[1]-1, valinit=0, valfmt='%d')
slider_step.on_changed(update_step)

plt.show()
print(segmentation[0,:,:].shape)

fig, ax = plt2.subplots()
plt2.title("Segs")
plt.subplots_adjust(bottom=0.15)
ax.imshow(segmentation[0,:,:])
ax_seg = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_seg = Slider(ax_seg, 'Step No.', 0, data.shape[1]-1, valinit=0, valfmt='%d')
slider_seg.on_changed(update_seg)

plt2.show()
