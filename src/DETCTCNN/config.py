from enum import Enum

H5_KEYS = ["dataH", "dataL", "label", "mfweights", "bweights", "cnr"] #affine, orig_shape, cnr

class ModelType(Enum):
    UNet_MINet = 1
    UNet_MCNet = 2
    UNet_LCNet = 3
    UNet_SECT = 4
    UNet_PredictFusion = 5
    DeepSynNet = 6


class DataAugType(Enum):
    NoImgDataAug = 0
    ImgAug = 1
    PhyAug = 2
    CNRAug = 3
    FeatureAug = 4

class BoundaryType(Enum):
    NoBoundary=0
    CNR=1
    LoG=2

class ResType(Enum):
    NoRes=1
    Res=2
    Dense=3
