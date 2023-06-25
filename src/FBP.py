from skimage.transform import iradon
import numpy as np
from matplotlib import pyplot as plt
import h5py


sinogram_path="../MUSIC2D_HDF5/sample10/fullSpectrum/projections/sinogram.h5"
image_path="../MUSIC2D_HDF5/sample10/fullSpectrum/reconstruction/reconstruction.h5"

rad_val = 60

dataFile = h5py.File(sinogram_path,"r")
with dataFile as f:
    data = np.array(f['data']['value'], order='F').transpose()
    print(data.shape)
    sinogram = data[rad_val,:,:]
    #sinogram = sinogram.squeeze(axis=0)
    dataFile.close()

dataFile = h5py.File(image_path,"r")
with dataFile as f:
    data = np.array(f['data']['value'], order='F').transpose()
    image = data[:,:,0,rad_val]
    dataFile.close()

theta = np.linspace(0., 180., max(sinogram.shape), endpoint=False)

reconstruction_fbp = iradon(sinogram, filter_name='ramp')


imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax2.set_title("Original")
ax2.imshow(image, cmap=plt.cm.Greys_r)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

plt.show()