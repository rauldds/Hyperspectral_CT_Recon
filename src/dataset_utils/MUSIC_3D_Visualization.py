import argparse

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import open3d as o3d
from collections import Counter
import os

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dataset", help="dataset path", type=str)

args = argParser.parse_args()
print("DATASET PATH: %s" % args.dataset)
# Path to the dataset
DATASET_PATH = args.dataset

if str(DATASET_PATH) == "None":
    print("PLEASE PROVIDE THE DATASET PATH")
    exit()
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

'''with h5py.File(DATASET_PATH+ "/"+file+'/manualSegmentation/manualSegmentation.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F').transpose()
print(data.shape)

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

num_classes = len(Counter(classes).keys())
#print(Counter(classes).keys())

colors = []
for i in range (num_classes):
    color = list(np.random.choice(range(256), size=3))
    colors.append(color)
colors =(np.asarray(colors))/255
point_colors = []

for idx in classes:
    point_colors.append(colors[int(idx)-1])
point_colors=np.asarray(point_colors)

#Conversion of point cloud to o3d format
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(points[:,:])
pcl.colors = o3d.utility.Vector3dVector(point_colors)

o3d.visualization.draw_geometries([pcl],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[int(data.shape[0]/2), 
                                          int(data.shape[1]/2), 
                                          int(data.shape[2]/2)],
                                  up=[0.2304, -0.8825, 0.4101],
                                  window_name="Manual Segmentation 3D Visualization")

'''
with h5py.File(DATASET_PATH + "/" + file
                + '/fullSpectrum/reconstruction/reconstruction.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F')
    print(data.shape)


print(f'sinogram shape: {data.shape}')

fig, ax = plt.subplots()
plt.title("Slices")
plt.subplots_adjust(bottom=0.15)
ax.imshow(data[0][0,:,:])
idx = 0
step = 0
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

ax_energy = plt.axes([0.25, 0.05, 0.5, 0.03])
slider_energy = Slider(ax_energy, 'Sinogram No.', 0, 127, valinit=0, valfmt='%d')
slider_energy.on_changed(update_energy_level)

ax_step = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_step = Slider(ax_step, 'Step No.', 0, 36, valinit=0, valfmt='%d')
slider_step.on_changed(update_step)

plt.show()

print(f'idx val: {idx}')