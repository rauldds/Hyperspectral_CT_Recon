"""Dataset Base Class"""

from abc import ABC, abstractmethod
from email.mime import image
import os
import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
# from music_2d_labels import MUSIC_2D_LABELS
from  src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS

class Dataset(ABC):
    def __init__(self, root, transform, partition, spectrum):
        self.root_path = root
        self.transform = transform
        self.partition = partition
        self.spectrum = spectrum

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""

# Data shape is (100,100,1,128). This is 2D data with 128 channels!
# TODO: Do we want to retrieve sinograms?
class MUSIC2DDataset(Dataset):
    def __init__(self, *args, root=None, transform=None, partition="train", spectrum="fullSpectrum", **kwargs):
        super().__init__(*args, root=root, transform=transform, partition=partition, spectrum=spectrum,
                            **kwargs)
        self.images = []
        self.classes = []
        self.segmentations = []
        #Collect all the class names
        for label in MUSIC_2D_LABELS:
            self.classes.append(label)
        self._load_data()
            
    
    def __len__(self):
        return len(self.images) 

    def _get_image(self,index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _get_segmentation(self,index):
        segmentation = self.segmentations[index]
        if self.transform is not None:
            segmentation = self.transform(segmentation)
        return segmentation
    
    def _get_classes(self, segmentation):
        uniques = np.unique(segmentation)
        #classes = []
        #for label in MUSIC_2D_LABELS:
        #    if MUSIC_2D_LABELS[label] in uniques:
        #        classes.append(label)
        return uniques

    def __getitem__(self, index):
        image = self._get_image(index=index)
        segmentation = self._get_segmentation(index=index) 
        classes = self._get_classes(segmentation)
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "segmentation": segmentation, "classes":classes}
    
    def plot_item(self,index, rad_val):
        image = self.images[index].squeeze()[:,:,rad_val]
        plt.title("Reconstruction\nFiltered back projection")
        plt.imshow(image.squeeze(), cmap=plt.cm.Greys_r)
        plt.show()

    def get_classes(self):
        return self.classes

    def _load_data(self):
        for path in os.listdir(self.root_path):
            if self.partition =="train" and (path == "sample20" or 
                                             path == "sample19" or
                                             path == "README.md"):
                continue
            elif self.partition == "valid" and (path != "sample19"):
                continue
            elif self.partition == "test" and (path != "sample20"):
                continue
            #TODO: Probably good to start with reduced spectrum instead of fullspectrum
            data_path = os.path.join(self.root_path, path, self.spectrum, "reconstruction")
            segmentation_file = h5py.File(os.path.join(self.root_path, path, "manualSegmentation", "manualSegmentation_global.h5"))
            # Open reconstructions
            reconstruction_file = None
            if os.path.isfile(os.path.join(data_path, "reconstruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"),"r")
            if os.path.isfile(os.path.join(data_path, "recontruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "recontruction.h5"),"r")
            #Collect image list
            with reconstruction_file as f:
                data = np.array(f['data']['value'], order='F')
                if self.spectrum=="fullSpectrum":
                    data = data.squeeze(1)
                self.images.append(data)
                reconstruction_file.close()
            with segmentation_file as f:
                data = np.array(f['data']['value'], order='F')
                self.segmentations.append(data.astype(int))
                segmentation_file.close()

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", help="dataset path", type=str, default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5")
    args = argParser.parse_args()
    DATASET_PATH = args.dataset
    print(DATASET_PATH)
    dataset = MUSIC2DDataset(root=DATASET_PATH,spectrum="reducedSpectrum",partition="valid")
    print(np.asarray(dataset[:]["segmentation"]).shape)