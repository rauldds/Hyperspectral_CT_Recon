"""Dataset Base Class"""
import pickle
from abc import ABC, abstractmethod
import argparse
import os
import random
import h5py
import numpy as np
from src.MUSIC_DATASET.utils import MUSIC_2D_LABELS
from src.MUSIC_DATASET.utils import EMPTY_SCANS
from src.OneD.config import hparams_LogReg
import torch


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
class MUSIC1DDataset(Dataset):
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
        split_location = "/Users/davidg/Projects/Hyperspectral_CT_Recon/splits/four_one_split.pkl"
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

    def __getitem__(self, index):
        return {"image": self.images[index], "segmentation": self.segmentations[index]}

    def _load_data(self):
        for path in os.listdir(self.path2d):
            if (self.partition == "all" or self.split_file) and path == "README.md":
                continue
            elif (self.partition == "all" or self.split_file):
                pass
            elif self.partition == "train" and (path == "sample20" or
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
            # print(path)
            # TODO: Probably good to start with reduced spectrum instead of fullspectrum
            data_path = os.path.join(self.path2d, path, self.spectrum, "reconstruction")
            segmentation_file = h5py.File(os.path.join(self.path2d, path, "manualSegmentation",
                                                       "manualSegmentation_global.h5"))
            # Open reconstructions
            reconstruction_file = None
            if os.path.isfile(os.path.join(data_path, "reconstruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"), "r")
            if os.path.isfile(os.path.join(data_path, "recontruction.h5")):
                reconstruction_file = h5py.File(os.path.join(data_path, "recontruction.h5"), "r")
            # Collect image list
            with reconstruction_file as f:
                data = np.array(f['data']['value'], order='F')
                if self.spectrum == "fullSpectrum":
                    data = data.squeeze(1)
                # Apply dimensionality reduction method to hyperspectral channels
                if self.band_selection is not None:
                    data = data[self.band_selection]
                # data = dimensionality_reduction(data, self.dim_red, data.shape, self.no_dim_red)
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
        if self.full_dataset and (self.partition == "test3D"):
            test_samples = ["Sample_23012018", "Sample_24012018"]
            idx = random.randint(0, 1)
            for path in os.listdir(self.path3d):
                if (path != test_samples[idx]):
                    continue

                data_path = os.path.join(self.path3d, path, self.spectrum, "reconstruction")
                # Open reconstructions
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"), "r")
                with reconstruction_file as f:
                    data = np.array(f['data']['value'], order='F')
                    for i in range(data.shape[1]):
                        scan = None
                        if self.spectrum == "reducedSpectrum":
                            scan = data[0:10, i, :, :]
                        else:
                            scan = data[:, i, :, :]
                        if self.band_selection is not None:
                            scan = scan[self.band_selection]
                        # scan = dimensionality_reduction(scan, self.dim_red, scan.shape, self.no_dim_red)
                        scan = torch.from_numpy(scan).float()
                        self.images.append(scan)
                        # HAD TO DO THIS BECAUSE NUMBER OF SEGMENTATION SLICES DOESN'T COINCIDE WITH THE NUMBER OF SCANS
                        self.segmentations.append(torch.zeros((100, 100)))

        if self.full_dataset and (self.partition == "train" or self.partition == "valid" or self.partition == "all"):
            upper_lim = 10
            limits = [0, upper_lim]
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
                reconstruction_file = h5py.File(os.path.join(data_path, "reconstruction.h5"), "r")
                # Collect image list
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
                        scan = data[:, i, :, :]
                        if self.band_selection is not None:
                            scan = scan[self.band_selection]
                        # scan = dimensionality_reduction(scan, self.dim_red, scan.shape, self.no_dim_red)
                        scan = torch.from_numpy(scan).float()
                        self.images.append(scan)
                with segmentation_file as f:
                    # print(path)
                    # empty_elements = []
                    data = np.array(f['data']['value'], order='F', dtype=np.int16)
                    data = torch.from_numpy(data).float()
                    if self.eliminate_empty == True and (path in EMPTY_SCANS):
                        data = np.delete(data, EMPTY_SCANS[path], axis=0)
                    for i in range(limits[0], limits[1]):
                        # if (data[i,:,:].argmax(0)==0).all():
                        # empty_elements.append(i)
                        self.segmentations.append(data[i, :, :])
                    # dict_empty_elements[path] = empty_elements
        # Take images from right split
        if self.split_file and self.split_data != None:
            if self.partition == "train" or self.partition == "valid":
                idx = self.split_data[self.partition]
                self.images = [self.images[i] for i in idx]
                self.segmentations = [self.segmentations[i] for i in idx]
        self.images = torch.stack(self.images)
        self.images = self.images.permute(0, 2, 3, 1)
        self.images = self.images.reshape(-1, 128)
        self.images = self.images.unsqueeze(1)
        background = self.images
        randBackground = self.images
        # self.images shape: torch.Size([4370000, 1, 128])
        self.segmentations = torch.stack(self.segmentations).argmax(dim=1)
        self.segmentations = self.segmentations.reshape(-1)
        randBackgroundSeg = self.segmentations

        # Remove all zeros for faster training
        nonzero = torch.nonzero(self.segmentations > 0).squeeze()
        no_zero_samples = nonzero.shape[0] // hparams_LogReg["num_black_division_factor"]
        zero_idxs = torch.nonzero(self.segmentations == 0).squeeze()
        # print(f"zero indices shape {zero_idxs}")
        self.images = self.images[nonzero]
        background = background[zero_idxs]
        # print(f"background shape {background.shape}")
        random_background = torch.randperm(background.size(0))[:no_zero_samples]
        # 'randBackground' contains 16320 samples of black "background pixels"
        randBackground = randBackground[random_background]
        # print(f"self.images shape {self.images.shape}")
        # print(f"randBackground shape {randBackground.shape}")
        self.images = torch.cat((self.images, randBackground), dim=0)
        # print(f"concat self images with background shape {self.images.shape}")


        self.segmentations = self.segmentations[nonzero]
        randBackgroundSeg = randBackgroundSeg[random_background]
        self.segmentations = torch.cat((self.segmentations, randBackgroundSeg), dim=0)
        self.images = list(self.images)
        self.segmentations = list(self.segmentations)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset",
                           help="dataset path", type=str,
                           default="/Users/davidg")
    args = argParser.parse_args()
    DATASET2D_PATH = args.dataset + "/MUSIC2D_HDF5"
    DATASET3D_PATH = args.dataset + "/MUSIC3D_HDF5"

    dataset = MUSIC1DDataset(path2d=DATASET2D_PATH, path3d=DATASET3D_PATH,
                             spectrum="fullSpectrum", partition="train", full_dataset=True)
    print(dataset[0]['image'].shape)
    print(len(dataset))