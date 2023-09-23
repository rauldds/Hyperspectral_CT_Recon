from abc import ABC, abstractmethod
import torch
import argparse
from src.DETCTCNN.data.music_2d_dataset import JointTransform2D, MUSIC2DDataset
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS
from src.OneD.config import hparams_LogReg

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

class MUSIC1DDataset(Dataset):
    '''
        This class loads the MUSIC dataset as a 1D Task.
    	Parameters:
            path2d (str): Path to the MUSIC2D_HDF5 folder
            path3d (str): Path to the MUSIC3D_HDF5 folder
            transform (function): transformations to be applied to the samples
            full_dataset (bool): load MUSIC3D 
            partition (str): which split to load
            spectrum (fullSpectrum/reducedSpectrum): load 10 or 128 energy version of dataset
            dim_red (str): which dimensionality reduction technique to use
            no_dim_red (int): How many hyperspectral bands to produce
            eliminate_empty (bool): Whether to eliminate empty scans or not.
            band_selection (str): Path pointing to pickle file containing selected
            bands
            include_nonthreat (bool): Include the NonThreat sample.
            oversample_2D (int): Applies oversampling to MUSIC2D
            split_file (str): Path to split file
    	Returns:
			The MUSIC dataset 
	'''
    def __init__(self, *args, path2d=None, path3d=None, 
                transform=None, full_dataset=False, partition="train", 
                spectrum="fullSpectrum", dim_red=None, no_dim_red=10, eliminate_empty=True, band_selection = None,
                include_nonthreat=True, oversample_2D=1, split_file=False, **kwargs):
        super().__init__(*args, path2d=path2d, path3d=path3d,
                         transform=transform, partition=partition,
                         spectrum=spectrum, full_dataset=full_dataset, **kwargs)
        self.images = []
        self.segmentations = []
        self.classes = []
        self.dim_red = dim_red
        self.path2d = path2d
        self.path3d = path3d
        self.spectrum = spectrum
        self.partition = partition
        self.eliminate_empty = eliminate_empty
        # Collect all the class names
        for label in MUSIC_2D_LABELS:
            self.classes.append(label)
        self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"image": self.images[index], "segmentation": self.segmentations[index]}

    def _load_data(self):

        jt = JointTransform2D((100,100),0,None,0,True,None,True)
        MD2D = MUSIC2DDataset(
            path2d=self.path2d, path3d=self.path3d, 
            spectrum=self.spectrum, 
            partition=self.partition,
            full_dataset=True, 
            transform=jt,
            dim_red=self.dim_red,
            no_dim_red=10,
            eliminate_empty=self.eliminate_empty,
            oversample_2D=1,
            split_file=False
        )
        self.images = MD2D[:]["image"]
        self.segmentations = MD2D[:]["segmentation"]

        self.images = torch.stack(self.images)
        self.images = self.images.permute(0, 2, 3, 1)
        print(self.images.shape)
        self.images = self.images.reshape(-1, 10)
        self.images = self.images.unsqueeze(1)
        background = self.images
        randBackground = self.images
        print(self.images.shape)
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
                           default="/media/rauldds/TOSHIBA EXT/MLMI")
    args = argParser.parse_args()
    DATASET2D_PATH = args.dataset + "/MUSIC2D_HDF5"
    DATASET3D_PATH = args.dataset + "/MUSIC3D_HDF5"

    dataset = MUSIC1DDataset(path2d=DATASET2D_PATH, path3d=DATASET3D_PATH,
                             spectrum="fullSpectrum", partition="train", full_dataset=True)
    print(dataset[0]['image'].shape)
    print(len(dataset))