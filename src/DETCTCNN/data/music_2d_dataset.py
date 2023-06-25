"""Dataset Base Class"""

from abc import ABC, abstractmethod
import os
import h5py
from matplotlib import pyplot as plt
import numpy as np


class Dataset(ABC):
    def __init__(self, root):
        self.root_path = root

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""

# Data shape is (100,100,1,128). This is 2D data with 128 channels!
# TODO: Do we want to retrieve sinograms?
class MUSIC2DDataset(Dataset):
    def __init__(self, *args, root=None, **kwargs):
        super().__init__(*args, root=root,
                            **kwargs)
        self.images = []
        self.sinograms = []
        self.classes = []
        # If we need any transformations
        self.transform = None
        for path in os.listdir(self.root_path):
            #Collect all the class names
            if "sample" not in path:
                self.classes.append(path)
            if "README" in path:
                continue
            data_path = os.path.join(self.root_path, path, "fullSpectrum", "reconstruction")
            # sinogram_path = os.path.join(self.root_path, path, "fullSpectrum", "projections", "sinogram.h5")
            # Open reconstructions
            reconstruction_file = None
            if os.path.isfile(os.path.join(data_path, "reconstruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"),"r")
            if os.path.isfile(os.path.join(data_path, "recontruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "recontruction.h5"),"r")
            # sinogram_file = h5py.File(sinogram_path,"r") 
            #Collect image list
            with reconstruction_file as f:
                data = np.array(f['data']['value'], order='F').transpose()
                self.images.append(data)
                reconstruction_file.close()
            # with sinogram_file as f:
            #     data = np.array(f['data']['value'], order='F').transpose()
            #     self.sinograms.append(data)
            #     sinogram_file.close()
            
    
    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def plot_item(self,index, rad_val):
        image = self.images[index].squeeze()[:,:,rad_val]
        plt.title("Reconstruction\nFiltered back projection")
        plt.imshow(image.squeeze(), cmap=plt.cm.Greys_r)
        plt.show()

if __name__ == "__main__":
    path = "/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5"
    dataset = MUSIC2DDataset(root=path)
    dataset.plot_item(30, 30)
    