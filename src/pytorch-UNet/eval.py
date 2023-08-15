import os
import torch

import torch.optim as optim

from functools import partial
from argparse import ArgumentParser

from unet.unet import UNet2D
from unet.model import Model
from unet.utils import MetricList
from unet.metrics import FocalLoss, jaccard_index, f1_score, LogNLLLoss, DiceLoss, miou
from unet.dataset import JointTransform2D, ImageToImage2D, Image2D
from unet.music_2d_dataset import MUSIC2DDataset

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default="./MUSIC2D_HDF5")
parser.add_argument('--checkpoint_path', type=str, default="./output")
parser.add_argument('--device', default='mps', type=str)
parser.add_argument('--out_channels', default=16, type=int)
parser.add_argument('--depth', default=4, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--save_model', default=0, type=int)
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--crop', type=int, default=20)
parser.add_argument("-sp", "--spectrum", type=str, default="reducedSpectrum", help="Spectrum of MUSIC dataset")
parser.add_argument("-dim_red", "--dim_red", choices=['none', 'pca'], default="none", help="Use dimensionality reduction")
parser.add_argument("-no_dim_red", "--no_dim_red", type=int, default=5, help="Target no. dimensions for dim reduction")
parser.add_argument('--chk_path', default="checkpoint/model/0500/model.pt", type=str)
args = parser.parse_args()

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_test = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)

test_dataset = MUSIC2DDataset(
    path2d=args.dataset, path3d=None,
    partition="test",
    spectrum=args.spectrum, 
    transform=None, 
    full_dataset=False,
    dim_red = args.dim_red, 
    no_dim_red = args.no_dim_red
)
in_channels = 10
if args.spectrum == "fullSpectrum":
    in_channels = 128
if args.dim_red != "none":
    in_channels = args.no_dim_red

conv_depths = [int(args.width*(2**k)) for k in range(args.depth)]
# unet = UNet2D(in_channels, args.out_channels, conv_depths)
unet = torch.load(args.chk_path)
unet.eval()
optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)
checkpoint = torch.load(args.chk_path)
loss = LogNLLLoss()

results_folder = os.path.join(args.checkpoint_path, args.model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

metric_list = MetricList({'jaccard': partial(miou),
                          'f1': partial(f1_score)})

model = Model(unet, loss, optimizer, results_folder, device=args.device)

model.predict_dataset(test_dataset, "./output", patch_size=20)