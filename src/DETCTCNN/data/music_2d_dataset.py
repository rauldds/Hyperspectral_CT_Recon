"""Dataset Base Class"""

from abc import ABC, abstractmethod
from email.mime import image
import torch.nn.functional as F2
import os
import pickle
import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
from src.DETCTCNN.data.data_utils import dimensionality_reduction
from  src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS
from  src.DETCTCNN.data.empty_scans import EMPTY_SCANS
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchio as tio
from torchvision import transforms as T
from torchvision.transforms import functional as F
import json
import random


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
                spectrum="fullSpectrum", dim_red=None, no_dim_red=10, eliminate_empty=True, band_selection = None,
                include_nonthreat=True, oversample_2D=1, split_file=False, **kwargs):

        super().__init__(*args, path2d=path2d, path3d=path3d,
                         transform=transform, partition=partition, 
                         spectrum=spectrum, full_dataset=full_dataset, **kwargs)
        self.images = []
        self.classes = []
        self.segmentations = []
        self.dim_red = dim_red
        self.no_dim_red = no_dim_red
        self.eliminate_empty =eliminate_empty
        self.band_selection = None
        self.include_nonthreat = include_nonthreat
        self.oversample_2D = oversample_2D
        self.split_file = split_file
        split_location = "./splits/four_one_split.pkl"
        self.split_data = None
        if split_file:
            with open(split_location, 'rb') as handle:
                self.split_data = pickle.load(handle)
        if band_selection:
            bands = pickle.load(open(band_selection, "rb"))
            self.band_selection = bands
        #Collect all the class names
        for label in MUSIC_2D_LABELS:
            self.classes.append(label)
        self._load_data()
            
    
    def __len__(self):
        return len(self.images) 

    def _get_image(self,index):
        image = self.images[index]
        return image

    def _get_segmentation(self,index):
        segmentation = self.segmentations[index]
        return segmentation

    def _patchify(self,data):
        print("hi")
    
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
        if self.transform is not None:
            image, segmentation = self.transform(image,segmentation)
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
            if (self.partition == "all" or self.split_file) and path == "README.md":
                continue
            elif (self.partition == "all" or self.split_file):
                pass
            elif self.partition =="train" and (path == "sample20" or
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
            elif self.partition == "test3D":
                continue
            #print(path)
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
                # Apply dimensionality reduction method to hyperspectral channels
                if self.band_selection is not None:
                    data = data[self.band_selection]
                data = dimensionality_reduction(data, self.dim_red, data.shape, self.no_dim_red)
                data = torch.from_numpy(data).float()
                for i in range(self.oversample_2D):
                    self.images.append(data)
                reconstruction_file.close()
            with segmentation_file as f:
                data = np.array(f['data']['value'], order='F')
                data = torch.from_numpy(data).float()
                for i in range(self.oversample_2D):
                    self.segmentations.append(data)
                segmentation_file.close()
        if self.full_dataset and (self.partition=="test3D"):
            test_samples = ["Sample_23012018", "Sample_24012018"]
            idx = random.randint(0,1)
            for path in os.listdir(self.path3d):
                if (path != test_samples[idx]):
                    continue

                data_path = os.path.join(self.path3d, path, self.spectrum, "reconstruction")
                # Open reconstructions
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"),"r")
                with reconstruction_file as f:
                    data = np.array(f['data']['value'], order='F')
                    for i in range(data.shape[1]):
                        scan = None
                        if self.spectrum == "reducedSpectrum":
                            scan = data[0:10, i, :, :]
                        else:
                            scan = data[:,i, :,:]
                        if self.band_selection is not None:
                            scan = scan[self.band_selection]
                        scan = dimensionality_reduction(scan, self.dim_red, scan.shape, self.no_dim_red)
                        scan = torch.from_numpy(scan).float()
                        self.images.append(scan)
                        # HAD TO DO THIS BECAUSE NUMBER OF SEGMENTATION SLICES DOESN'T COINCIDE WITH THE NUMBER OF SCANS
                        self.segmentations.append(torch.zeros((100,100)))

        if self.full_dataset and (self.partition=="train" or self.partition=="valid" or self.partition=="all"):
            upper_lim = 10
            limits = [0,upper_lim]
            # dict_empty_elements = {}
            for path in os.listdir(self.path3d):
                # TODO: SHORTER WAY TO DO THIS
                # with except of README, the rest of folders below don't have a correct correspondence
                # between the number of slices and the number of segmentations
                if (path == "README.md" or path == "Fruits" or
                    path == "Sample_23012018" 
                    # or path == "Sample_24012018"
                    or (not self.include_nonthreat and path == "NonThreat")    
                ):
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
                    if self.eliminate_empty == True and (path in EMPTY_SCANS):
                        data = np.delete(data, EMPTY_SCANS[path], axis=1)
                    if self.partition == "train":
                        limits = [upper_lim, data.shape[1]]
                    # Get all data for custom split file
                    if self.partition == "all" or self.split_file:
                        limits = [0, data.shape[1]]
                    # TODO: Might be a more optimal way to do this hehe
                    for i in range(limits[0], limits[1]):
                        scan = data[:,i, :,:]
                        if self.band_selection is not None:
                            scan = scan[self.band_selection]
                        scan = dimensionality_reduction(scan, self.dim_red, scan.shape, self.no_dim_red)
                        scan = torch.from_numpy(scan).float()
                        self.images.append(scan)
                with segmentation_file as f:
                    #print(path)
                    #empty_elements = []
                    data = np.array(f['data']['value'], order='F',dtype=np.int16)
                    data = torch.from_numpy(data).float()
                    if self.eliminate_empty == True and (path in EMPTY_SCANS):
                        data = np.delete(data, EMPTY_SCANS[path], axis=0)
                    for i in range(limits[0], limits[1]):
                        #if (data[i,:,:].argmax(0)==0).all():
                            #empty_elements.append(i)
                        self.segmentations.append(data[i, :, :])
                    #dict_empty_elements[path] = empty_elements
        # Take images from right split
        if self.split_file and self.split_data != None:
            if self.partition == "train" or self.partition == "valid":
                idx = self.split_data[self.partition]
                self.images = [self.images[i] for i in idx]
                self.segmentations = [self.segmentations[i] for i in idx]


                

class MusicTransform:
    def __init__(self, resize=128):
        self.resize = resize
        self.aug = A.Compose([
        # A.CenterCrop(85,85),
        A.Resize(resize,resize),
        # A.RandomRotate90(),
        # A.Affine(),
        # A.GaussNoise(var_limit=(0.01,0.1)),
        ToTensorV2(),
    ])
      
    def __call__(self, img):
        img = np.array(img)
        img = img.transpose((1,2,0))
        return self.aug(image=img)["image"]

class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    
    def __init__(self, crop=(96, 96), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False, resize=None, erosion=False):
        self.crop = crop
        self.erosion = erosion
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        self.resize = resize
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop

        if self.crop:
            indices = torch.nonzero((mask.argmax(0) != 0))
            # Do Smart cropping
            if indices.nelement() != 0:
                idx = random.randint(0,len(indices)-1)
                center = indices[idx]
                top = max(0,int(center[0]-self.crop[0]/2))
                left = max(0,int(center[1]-self.crop[0]/2))
                bottom = min(100,int(center[0]+self.crop[0]/2))
                right = min(100,int(center[1]+self.crop[0]/2))
                if top != 0:
                    if bottom == 100:
                        top = bottom - self.crop[0]
                if left != 0:
                    if right == 100:
                        left = right - self.crop[0]
            
                image, mask = F.crop(image, top, left, self.crop[0], self.crop[0]), F.crop(mask, top, left, self.crop[0], self.crop[0])
            else:
                # Do regular cropping
                i, j, h, w = T.RandomCrop.get_params(image, self.crop)
                image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
            
        if self.erosion:
            #print(image.shape)
            #print(image.unsqueeze(1).shape)
            kernel = torch.ones(1, 1, 3, 3).to(image.device)
            #print(kernel.shape)
            kernel[0,0,1,1]=0
            smoothed_img = F2.conv2d(image.unsqueeze(1), kernel, padding=1)
            #print(smoothed_img.shape)
            image = smoothed_img.squeeze(1)
            #print(image.shape)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        if self.resize:
            image, mask = F.resize(image, size=self.resize), F.resize(mask, size=self.resize)
        # transforming to tensor
        return image, mask

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
    
