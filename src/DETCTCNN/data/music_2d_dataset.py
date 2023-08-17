"""Dataset Base Class"""

from abc import ABC, abstractmethod
from email.mime import image
import os
import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
from data_utils import dimensionality_reduction
from  src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchio as tio
from torchvision import transforms as T
from torchvision.transforms import functional as F


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
                spectrum="fullSpectrum", dim_red=None, no_dim_red=10, **kwargs):
        super().__init__(*args, path2d=path2d, path3d=path3d,
                         transform=transform, partition=partition, 
                         spectrum=spectrum, full_dataset=full_dataset, **kwargs)
        self.images = []
        self.classes = []
        self.segmentations = []
        self.dim_red = dim_red
        self.no_dim_red = no_dim_red
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
                # Apply dimensionality reduction method to hyperspectral channels
                data = dimensionality_reduction(data, self.dim_red, data.shape, self.no_dim_red)
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
                    data = dimensionality_reduction(data, self.dim_red, data.shape, self.no_dim_red)
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
                             spectrum="reducedSpectrum", partition="train",full_dataset=False)
    #print(dataset[:]["classes"])
    print(len(dataset[:]["image"]))
    print(len(dataset[:]["segmentation"]))
    print(dataset[0]["image"].shape)
    print(dataset[0]["segmentation"].shape)
    #print(np.argmax(dataset[0]["segmentation"],0).unique())
    dataset_list = []
    for data in (dataset):
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=data["image"].unsqueeze(0)),
            segmentation=tio.LabelMap(tensor=torch.argmax(data["segmentation"],0).unsqueeze(0).repeat(1,10,1,1)),
        )
        dataset_list.append(subject)
    SubjectDataset = tio.data.SubjectsDataset(dataset_list)
    sampler = tio.GridSampler(patch_size=(10, 50, 50),subject=subject)
    patches_queue = tio.Queue(
        SubjectDataset,
        max_length=300,
        samples_per_volume=2,
        sampler=sampler,
        num_workers=1,
    )

    patches_loader = torch.utils.data.DataLoader(
        patches_queue,
        batch_size=16,
        num_workers=0,  # this must be 0
    )

    for patches_batch in patches_loader:
        inputs = patches_batch['image'][tio.DATA]  # key 't1' is in subject
        targets = patches_batch['segmentation'][tio.DATA]  # key 'brain' is in subject
        print(targets.shape)

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
    def __init__(self, crop=(256, 256), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        return image, mask