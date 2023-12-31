''' 
    This script performs inference on a volume using 2D network
'''

from argparse import ArgumentParser
import numpy as np
from torch.utils.data import DataLoader
from src.DETCTCNN.model.model import get_model
import torch
import k3d
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import open3d as o3d
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from  src.DETCTCNN.data.music_2d_dataset import MUSIC2DDataset, JointTransform2D
from src.DETCTCNN.model.utils import calculate_min_max, calculate_data_statistics, standardize, normalize
LABELS_SIZE = len(MUSIC_2D_LABELS)
INPUT_CHANNELS ={
    "reducedSpectrum": 10,
    "fullSpectrum":128
}

fig, ax = plt.subplots()
volume = 0


def update_slice(val):
    ax.cla()
    ax.imshow(volume[int(val)])
    fig.canvas.draw_idle()

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)

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


def main(args):
    global volume
    # Access the dataset folders
    path2d = args.data_root + "/MUSIC2D_HDF5"
    path3d = args.data_root + "/MUSIC3D_HDF5"

    energy_levels = 10
    if args.spectrum != "reducedSpectrum":
        energy_levels = 128
    if args.dim_red != "none" or args.band_selection is not None:
        energy_levels = args.no_dim_red

    transform = JointTransform2D(crop=None, p_flip=0.0, color_jitter_params=None, long_mask=True,erosion=args.erosion)

    train_dataset = MUSIC2DDataset(
        path2d=path2d, path3d=path3d, 
        spectrum=args.spectrum, 
        partition="test3D",
        full_dataset=True, 
        transform=None,
        dim_red=args.dim_red,
        no_dim_red=args.no_dim_red,
        band_selection=args.band_selection
        )
    dataset = MUSIC2DDataset(
        path2d=path2d, path3d=path3d, 
        spectrum=args.spectrum, 
        partition="test3D",
        full_dataset=True, 
        transform=transform,
        dim_red=args.dim_red,
        no_dim_red=args.no_dim_red,
        band_selection=args.band_selection,
    )
    print(dataset.images[0].shape)
    if args.normalize_data:
        mean, std = calculate_data_statistics(train_dataset.images)
        print(mean.shape)
        dataset.images = list(map(lambda x: standardize(x,mean,std) , dataset.images))
        min, max  = calculate_min_max(train_dataset.images)
        dataset.images = list(map(lambda x: normalize(x,min,max) , dataset.images))

    model = get_model(input_channels=energy_levels, n_labels=args.n_labels, use_bn=True, basic_out_channel=32, depth=3, dropout=0.7)
    checkpoint = torch.load(args.checkpoint, 
    #model = get_model(input_channels=energy_levels, n_labels=args.n_labels, 
    #                  use_bn=True, basic_out_channel=16, depth=1, dropout=0.5)
    #checkpoint = torch.load("./model_bsnet30merge.pt", 
                            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    palette = np.array(MUSIC_2D_PALETTE)
    test_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        FIRST_BATCH_FLAG = 0
        for batch in test_loader:
            x = batch['image']
            x = model(x[:, :, 2:98, 2:98])
            pred = x.argmax(dim=1).detach().cpu().numpy()
            if FIRST_BATCH_FLAG == 0:
                volume = pred
                FIRST_BATCH_FLAG = 1
            else:
                volume = np.concatenate([volume, pred], axis=0)
        
        print(f'volume shape: {volume.shape}')
        print(f'classes in test: {np.unique(volume)}')
        points, classes = pointcloud_converter(volume)
        print(points.shape)
        
        #for value in classes:
        #    colored_image = palette[pred]
        #    colored_image = colored_image.astype(np.uint8)

        point_colors = np.asarray(palette[classes])
        print(point_colors.shape)
    
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl.colors = o3d.utility.Vector3dVector(point_colors/255)
    o3d.io.write_point_cloud("test.ply",pcl)
    o3d.visualization.draw_geometries_with_animation_callback([pcl],rotate_view,
                                  window_name="Material Segmentation Prediction 3D Visualization")
    
    volume = np.asarray(palette[volume])

    plt.title("Slices")
    plt.subplots_adjust(bottom=0.15)
    ax.imshow(volume[0])

    ax_energy = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_slice = Slider(ax_energy, 'Sinogram No.', 0, (volume.shape[0]-1), valinit=0, valfmt='%d')
    slider_slice.on_changed(update_slice)

    plt.show()
    
    color_map = []
    for count, color in enumerate(MUSIC_2D_PALETTE):
        color_i = list(map(lambda x: x/255.0, color))
        color_i.insert(0,count/15.0)
        color_map.append(color_i)
    print(color_map)

    volume_k3d = k3d.points(positions=points,
                          point_size=0.1,
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
    parser.add_argument("-dr", "--data_root", type=str, default=".", help="Data root directory")
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-n", "--normalize_data", type=bool, default=False, help="decide if you want to normalize the data")
    parser.add_argument("-sp", "--spectrum", type=str, default="fullSpectrum", help="Spectrum of MUSIC dataset")
    parser.add_argument("-dim_red", "--dim_red", choices=['none', 'pca', 'merge'], default="none", help="Use dimensionality reduction")
    parser.add_argument("-no_dim_red", "--no_dim_red", type=int, default=10, help="Target no. dimensions for dim reduction")
    parser.add_argument("-bsel", "--band_selection", type=str, default=None, help="path to band list")
    parser.add_argument("-e", "--erosion", type=bool, default=False, help="path to band list")
    parser.add_argument("-chk", "--checkpoint", type=str, default="model_ce_no_container.pt", help="path to band list")
    args = parser.parse_args()

    main(args)
