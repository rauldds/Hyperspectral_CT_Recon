"""Dataset Base Class"""

from abc import ABC, abstractmethod
from email.mime import image
import os
import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
from  src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Dataset(ABC):
    def __init__(self, path2d, path3d, transform, partition, spectrum, full_dataset):
        self.path2d = path2d
        self.path3d = path3d
        self.transform = transform
        self.partition = partition
        self.spectrum = spectrum
        self.full_dataset = full_dataset

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""

# Data shape is (100,100,1,128). This is 2D data with 128 channels!
# TODO: Do we want to retrieve sinograms?
class MUSIC2DDataset(Dataset):
    def __init__(self, *args, path2d=None, path3d=None, 
                transform=None, full_dataset=False, partition="train", 
                spectrum="fullSpectrum", **kwargs):
        super().__init__(*args, path2d=path2d, path3d=path3d,
                         transform=transform, partition=partition, 
                         spectrum=spectrum, full_dataset=full_dataset, **kwargs)
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
            image = image.unsqueeze(-1)
            image = self.transform(image)
            image = image.squeeze(-1)
        return image

    def _get_segmentation(self,index):
        segmentation = self.segmentations[index]
        if self.transform is not None:
            segmentation = segmentation.unsqueeze(-1)
            segmentation = self.transform(segmentation)
            segmentation = segmentation.squeeze(-1)
        return segmentation
    
    def _get_classes(self, segmentation):
        if not torch.is_tensor(segmentation):
            segmentation = torch.stack(segmentation)
            list_classes = []
            for i in range(segmentation.shape[0]):
                  classes = torch.zeros((len(self.classes)))
                  uniques = torch.unique(segmentation[i].argmax(0))
                  classes[uniques] = 1
                  list_classes.append(classes)
            classes = torch.stack(list_classes)
        else:         
            classes = torch.zeros((len(self.classes)))
            uniques = torch.unique(segmentation.argmax(0))
            classes[uniques] = 1
        return classes

    def __getitem__(self, index):
        image = self._get_image(index=index)
        segmentation = self._get_segmentation(index=index) 
        classes = self._get_classes(segmentation)
        return {"image": image, "segmentation": segmentation, "classes":classes}
    
    def plot_item(self,index, rad_val):
        image = self.images[index].squeeze()[rad_val]
        plt.title("Reconstruction\nFiltered back projection")
        plt.imshow(image.squeeze(), cmap=plt.cm.Greys_r)
        plt.show()

    def plot_segmentation(self, index):
        data = self.segmentations[index]
        data = data.argmax(axis=0)
        plt.imshow(data)
        plt.colorbar()
        plt.show()

    def get_classes(self):
        return self.classes

    def _load_data(self):
        for path in os.listdir(self.path2d):
            if self.partition =="train" and (path == "sample20" or
                                             path == "sample19" or
                                             path == "sample1" or
                                             path == "sample2" or
                                             path == "README.md"):
                continue
            elif self.partition == "valid" and not (path == "sample19" or
                                                path == "sample1"):
                continue
            elif self.partition == "test" and not (path == "sample20" or
                                               path == "sample2"):
                continue
            #TODO: Probably good to start with reduced spectrum instead of fullspectrum
            data_path = os.path.join(self.path2d, path, self.spectrum, "reconstruction")
            segmentation_file = h5py.File(os.path.join(self.path2d, path, "manualSegmentation",
                                                      "manualSegmentation_global.h5"))
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
                data = torch.from_numpy(data).float()
                self.images.append(data)
                reconstruction_file.close()
            with segmentation_file as f:
                data = np.array(f['data']['value'], order='F')
                data = torch.from_numpy(data).float()
                self.segmentations.append(data)
                segmentation_file.close()
        if self.full_dataset and (self.partition=="train"):
            for path in os.listdir(self.path3d):
                # TODO: SHORTER WAY TO DO THIS
                # with except of README, the rest of folders below don't have a right correspondence
                # between the number of slices and the number of segmentations
                if (path == "README.md" or path == "Fruits" or
                    path == "Sample_23012018" or path == "Sample_24012018"):
                    continue
                data_path = os.path.join(self.path3d, path, self.spectrum, "reconstruction")
                segmentation_file = h5py.File(os.path.join(self.path3d, path, 
                                            "manualSegmentation", 
                                            "manualSegmentation_global.h5"))
                # Open reconstructions
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"),"r")
                #Collect image list
                with reconstruction_file as f:
                    data = np.array(f['data']['value'], order='F')
                    data = torch.from_numpy(data).float()
                    # TODO: Might be a more optimal way to do this hehe
                    for i in range(data.shape[1]):
                        self.images.append(data[:, i, :, :])
                with segmentation_file as f:
                    data = np.array(f['data']['value'], order='F',dtype=np.int16)
                    data = torch.from_numpy(data).float()
                    for i in range(data.shape[0]):
                        self.segmentations.append(data[i, :, :])

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", 
                           help="dataset path", type=str, 
                           default="/Users/luisreyes/Courses/MLMI/Hyperspectral_CT_Recon/MUSIC2D_HDF5")
    args = argParser.parse_args()
    DATASET2D_PATH = "/media/rauldds/TOSHIBA EXT/MLMI/MUSIC2D_HDF5"
    DATASET3D_PATH = "/media/rauldds/TOSHIBA EXT/MLMI/MUSIC3D_HDF5"

    dataset = MUSIC2DDataset(path2d=DATASET2D_PATH, path3d=DATASET3D_PATH,
                             spectrum="reducedSpectrum", partition="train",full_dataset=True)
    #print(dataset[:]["classes"])
    print(len(dataset[:]["image"]))
    print(len(dataset[:]["segmentation"]))

class ImgAugTransform:
    def __init__(self):
        self.aug = A.Compose([
        A.RandomCrop(height=64, width=64),
        A.RandomRotate90(),
        A.Affine(p=0.5),
        ToTensorV2(),
    ])
      
    def __call__(self, img):
        img = np.array(img)
        img = img.transpose((1,2,0))
        return self.aug(image=img)["image"]