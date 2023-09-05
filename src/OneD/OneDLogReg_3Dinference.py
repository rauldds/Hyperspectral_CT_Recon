from argparse import ArgumentParser

from matplotlib import pyplot as plt
from OneDLogReg import OneDLogReg
import k3d
import torch
from src.MUSIC_DATASET import MUSIC2DDataset
# from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from src.MUSIC_DATASET.utils import MUSIC_2D_LABELS
from src.MUSIC_DATASET.utils import MUSIC_2D_PALETTE
from tqdm import tqdm
from utils import image_from_segmentation
import numpy as np
from torch.utils.data import DataLoader
import open3d as o3d

LABELS_SIZE = len(MUSIC_2D_LABELS)
palette = np.array(MUSIC_2D_PALETTE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = OneDLogReg().to(device=device)
checkpoint = torch.load("best_model.pth",
                        map_location=torch.device(device=device))
model.load_state_dict(checkpoint)
model.eval()

def pointcloud_converter(data):
    '''Function to convert the segmented slices into a point cloud for 3D visualization'''
    points = []
    classes = []
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                if data[x,y,z]!=0:
                    point = [[y],[z], [x]]
                    points.append(point)
                    classes.append(data[x,y,z])
    points = (np.asarray(points)).squeeze(2)
    return points, classes

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)

def main(args):

    global volume 
    path2d = args.data_root + "/MUSIC2D_HDF5"
    path3d = args.data_root + "/MUSIC3D_HDF5"
    dataset = MUSIC2DDataset(path2d=path2d, path3d=path3d, partition="test3D", spectrum="fullSpectrum",
                                transform=None, full_dataset=True)
    pred = torch.empty(size=(100,100, 16)).to(device)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    volume = torch.empty(len(dataset),100,100,16)
    # image = dataset[args.sample]["image"].permute(1,2,0).to(device)
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader)):
            image = batch["image"].squeeze(0).permute(1,2,0).to(device)
            for i in range(image.shape[0]):
                # Use column batch size
                pixel = image[i, :]
                pixel = pixel.unsqueeze(1)
                out = model(pixel)
                # out = out.argmax(dim=2)
                out = out.squeeze()
                # print(out)
                pred[i,:] = out
            volume[idx] = pred
        volume = volume.permute(3,0,1,2)
        volume = volume.argmax(0)
        print(f'volume shape: {volume.shape}')
        print(f'classes in test: {np.unique(volume)}')
        points, classes = pointcloud_converter(volume)

        point_colors = np.asarray(palette[classes])
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(point_colors/255)
        o3d.io.write_point_cloud("test.ply",pcl)
        o3d.visualization.draw_geometries_with_animation_callback([pcl],rotate_view,
                                    window_name="Material Segmentation Prediction 3D Visualization")

        color_map = []
        for count, color in enumerate(MUSIC_2D_PALETTE):
            color_i = list(map(lambda x: x/255.0, color))
            color_i.insert(0,count/15.0)
            color_map.append(color_i)
        print(color_map)
        
        volume_k3d = k3d.points(positions=points,
                          point_size=1,
                          shader='flat',
                          opacity=0.5,
                          color_map=color_map,
                          attribute=(np.array(classes)/15.0).tolist(),
                          name="3D VOLUME PLOT")
        plot = k3d.plot()
        plot += volume_k3d
        with open('test.html', 'w') as f:
            f.write(plot.get_snapshot())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dr", "--data_root", type=str, default='/home/davidge/Documents/Projects/Hyperspectral_CT_Recon', help="Data root directory")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-n", "--normalize_data", type=bool, default=False, help="decide if you want to normalize the data")
    parser.add_argument("-sp", "--spectrum", type=str, default="fullSpectrum", help="Spectrum of MUSIC dataset")
    parser.add_argument("-s", "--sample", type=int, default=8, help="Number of slices")
    parser.add_argument("-dim_red", "--dim_red", choices=['none', 'pca', 'merge'], default='none', help="Use dimensionality reduction")
    parser.add_argument("-no_dim_red", "--no_dim_red", type=int, default=128, help="Target no. dimensions for dim reduction")
    parser.add_argument("-bsel", "--band_selection", type=str, default="band_selection/band_sel_bsnet_norm_30_bands.pkl", help="path to band list")
    args = parser.parse_args()
    main(args)
