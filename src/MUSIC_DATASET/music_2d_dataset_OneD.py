"""Dataset Base Class"""

from abc import ABC, abstractmethod
import argparse
import os
import h5py
import numpy as np
from src.MUSIC_DATASET.utils import MUSIC_2D_LABELS
from src.MUSIC_DATASET.utils import EMPTY_SCANS
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
                 spectrum="fullSpectrum", dim_red=None, no_dim_red=10, band_selection=None, **kwargs):
        super().__init__(*args, path2d=path2d, path3d=path3d,
                         transform=transform, partition=partition,
                         spectrum=spectrum, full_dataset=full_dataset, **kwargs)
        self.images = []
        self.segmentations = []
        self.classes = []
        self.dim_red = dim_red
        self.no_dim_red = no_dim_red
        # Collect all the class names
        for label in MUSIC_2D_LABELS:
            self.classes.append(label)
        self._load_data()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return {"image": self.images[index], "segmentation": self.segmentations[index]}

    def _load_data(self):
        for path in os.listdir(self.path2d):
            if self.partition == "train" and (path == "sample20" or
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
                data = torch.from_numpy(data).float()
                self.images.append(data)
                reconstruction_file.close()
            with segmentation_file as f:
                data = np.array(f['data']['value'], order='F')
                data = torch.from_numpy(data).float()
                self.segmentations.append(data)
                segmentation_file.close()
        if self.full_dataset and (self.partition == "train" or self.partition == "valid"):
            upper_lim = 10
            limits = [0, upper_lim]
            # dict_empty_elements = {}
            for path in os.listdir(self.path3d):
                if (path == "README.md" or path == "Fruits" or
                        path == "Sample_23012018" or path == "Sample_24012018"):
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
                    data = torch.from_numpy(data).float()
                    data = np.delete(data, EMPTY_SCANS[path], axis=1)
                    if self.partition == "train":
                        limits = [upper_lim, data.shape[1]]
                    # TODO: Might be a more optimal way to do this hehe
                    for i in range(limits[0], limits[1]):
                        self.images.append(data[:, i, :, :])
                with segmentation_file as f:
                    # print(path)
                    # empty_elements = []
                    data = np.array(f['data']['value'], order='F', dtype=np.int16)
                    data = torch.from_numpy(data).float()
                    data = np.delete(data, EMPTY_SCANS[path], axis=0)
                    for i in range(limits[0], limits[1]):
                        # if (data[i,:,:].argmax(0)==0).all():
                        # empty_elements.append(i)
                        self.segmentations.append(data[i, :, :])
        self.images = torch.stack(self.images)
        self.images = self.images.permute(0, 2, 3, 1)
        self.images = self.images.reshape(-1, 128)
        self.images = self.images.unsqueeze(1)
        self.images = self.images.unsqueeze(2)
        self.segmentations = torch.stack(self.segmentations).argmax(dim=1)
        self.segmentations = self.segmentations.reshape(-1)
        # Remove all zeros for faster training
        nonzero = torch.nonzero(self.segmentations > 0).squeeze()
        self.images = self.images[nonzero]
        self.segmentations = self.segmentations[nonzero]
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
    # print(dataset[:]["classes"])