import numpy as np
import h5py
import os

# Path to the dataset
DATASET_PATH = '/media/davidg-dl/Second SSD/MUSIC2D_HDF5/'

# Uncomment if the .txt files with the class numbers hast been generated
# IMPORTANT: a README.md is also written to the file.txt it needs to be manually deleted.
# classes = os.listdir('/media/davidg-dl/Second SSD/MUSIC2D_HDF5')
# with open('[MUSIC2D]class_names.txt', 'w')as f:
#     for cls in classes:
#         f.write(cls+"\n")
# f.close()

# Print all classes to console
def get_classes(file_path=None):
    with open(file_path, 'r') as f:
        classes = f.read().splitlines()
        return classes
classes = get_classes('/home/davidg-dl/Documents/MLMI/DatasetVisualization/[MUSIC2D]class_names.txt')

print(f'[INFO] please select the class to get a sample from: 0 - {len(classes) -1}')
for i, cls in enumerate(classes):
    print(f"{i} - {cls}")

class_selection = input("selection: ")
class_selection = int(class_selection)


# Manual Segmentation data visualization
with h5py.File(DATASET_PATH+classes[class_selection]+'/manualSegmentation/manualSegmentation.h5', 'r') as f:
    data = np.array(f['data']['value'], order='F').transpose()
    # print(data.shape)
    # data = data[127, :, :]
    # data = data.squeeze(axis=2)
    # data = data[:, :, 1]
#dataFile.close()

print(f'visualizing {classes[class_selection]} manual segmentation')
print(f'Located on {DATASET_PATH+classes[class_selection]}/manualSegmentation/manualSegmentation.h5')

from matplotlib import pyplot as plt
plt.imshow(data, interpolation='nearest')
plt.show()

