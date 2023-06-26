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
        self.classes = []
        self.segmentations = []
        # If we need any transformations
        self.transform = None
        for path in os.listdir(self.root_path):
            #Collect all the class names
            if "sample" not in path:
                self.classes.append(path)
            if "README" in path:
                continue
            data_path = os.path.join(self.root_path, path, "fullSpectrum", "reconstruction")
            segmentation_file = h5py.File(os.path.join(self.root_path, path, "manualSegmentation", "manualSegmentation.h5"))
            # Open reconstructions
            reconstruction_file = None
            if os.path.isfile(os.path.join(data_path, "reconstruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"),"r")
            if os.path.isfile(os.path.join(data_path, "recontruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "recontruction.h5"),"r")
            #Collect image list
            with reconstruction_file as f:
                data = np.array(f['data']['value'], order='F').transpose()
                self.images.append(data)
                reconstruction_file.close()
            with segmentation_file as f:
                data = np.array(f['data']['value'], order='F').transpose()
                self.segmentations.append(data)
                segmentation_file.close()
            
    
    def __len__(self):
        return len(self.images) 

    def _get_image(self,index):
        image = self.images[index]
        # TODO: What order of dimensions
        image = image.transpose((3,2,0,1))
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _get_segmentation(self,index):
        segmentation = self.segmentations[index]
        return segmentation
    
    def __getitem__(self, index):
        image = self._get_image(index=index)
        segmentation = self._get_segmentation(index=index) 
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "segmentation": segmentation}
    
    def plot_item(self,index, rad_val):
        image = self.images[index].squeeze()[:,:,rad_val]
        plt.title("Reconstruction\nFiltered back projection")
        plt.imshow(image.squeeze(), cmap=plt.cm.Greys_r)
        plt.show()

    def get_classes(self):
        return self.classes

if __name__ == "__main__":
    path = "/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5"
    dataset = MUSIC2DDataset(root=path)
    print(dataset[0]["image"].shape)
    