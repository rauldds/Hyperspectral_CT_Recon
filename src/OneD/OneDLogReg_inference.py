from argparse import ArgumentParser

from matplotlib import pyplot as plt
from OneDLogReg import OneDLogReg
import torch
from src.MUSIC_DATASET import MUSIC2DDataset
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from tqdm import tqdm
from utils import image_from_segmentation
import numpy as np

LABELS_SIZE = len(MUSIC_2D_LABELS)
palette = np.array(MUSIC_2D_PALETTE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = OneDLogReg().to(device=device)
checkpoint = torch.load("best_model.pth",
                        map_location=torch.device(device=device))
print(type(checkpoint))
model.load_state_dict(checkpoint)
model.eval()


def main(args):
    path2d = args.data_root + "/MUSIC2D_HDF5"
    path3d = args.data_root + "/MUSIC3D_HDF5"
    dataset = MUSIC2DDataset(path2d=path2d, path3d=path3d, partition="train", spectrum="fullSpectrum",
                                transform=None, full_dataset=True)
    image = dataset[args.sample]["image"].permute(1,2,0).to(device)
    pred = torch.empty(size=(100,100, 16)).to(device)
    for i in tqdm(range(image.shape[0])):
        # Use column batch size
        pixel = image[i, :]
        pixel = pixel.unsqueeze(1)
        out = model(pixel)
        # out = out.argmax(dim=2)
        out = out.squeeze()
        # print(out)
        pred[i,:] = out
    print(pred.shape)
    pred = pred.unsqueeze(0).permute((0, 3, 1, 2))
    image_from_segmentation(pred, LABELS_SIZE, MUSIC_2D_PALETTE, device=device, mode="train")
    # pred = pred.cpu().numpy().astype(np.uint8)
    # colored_pred = palette[pred]
    # print(colored_pred.shape)

    # plt.title("Prediction")
    # plt.imshow(colored_pred)
    # plt.axis('off')
    # plt.show()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default="/Users/davidg/Projects/Hyperspectral_CT_Recon", help="Data root directory")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-n", "--normalize_data", type=bool, default=False, help="decide if you want to normalize the data")
    parser.add_argument("-sp", "--spectrum", type=str, default="fullSpectrum", help="Spectrum of MUSIC dataset")
    parser.add_argument("-s", "--sample", type=int, default=256, help="NUmber of slices")
    parser.add_argument("-dim_red", "--dim_red", choices=['none', 'pca', 'merge'], default='none', help="Use dimensionality reduction")
    parser.add_argument("-no_dim_red", "--no_dim_red", type=int, default=128, help="Target no. dimensions for dim reduction")
    parser.add_argument("-bsel", "--band_selection", type=str, default="band_selection/band_sel_bsnet_norm_30_bands.pkl", help="path to band list")
    args = parser.parse_args()

    main(args)
