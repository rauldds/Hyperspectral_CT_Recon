from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import music_2d_dataset
from src.DETCTCNN.model.metrics import mIoU_score
from src.DETCTCNN.model.model import get_model
import torch
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from  src.DETCTCNN.data.music_2d_dataset import MUSIC2DDataset
from src.DETCTCNN.model.utils import calculate_min_max, calculate_data_statistics, standardize, normalize
LABELS_SIZE = len(MUSIC_2D_LABELS)
INPUT_CHANNELS ={
    "reducedSpectrum": 10,
    "fullSpectrum":128
}

def main(args):

    path2d = args.data_root + "/MUSIC2D_HDF5"
    path3d = args.data_root + "/MUSIC3D_HDF5"
    train_dataset = MUSIC2DDataset(path2d=path2d, path3d=path3d,partition="train",spectrum="reducedSpectrum", transform=None)
    transform = music_2d_dataset.JointTransform2D(long_mask=True, resize=(120,120), color_jitter_params=None, crop=None,p_flip=0)
    dataset = MUSIC2DDataset(path2d=path2d, path3d=path3d, 
                            spectrum=args.spectrum, partition="valid",
                            full_dataset=True,
                            transform=transform)
    if args.normalize_data:
        mean, std = calculate_data_statistics(train_dataset.images)
        dataset.images = list(map(lambda x: standardize(x,mean,std) , dataset.images))
        min, max  = calculate_min_max(train_dataset.images)
        dataset.images = list(map(lambda x: normalize(x,min,max) , dataset.images))
    model = get_model(input_channels=INPUT_CHANNELS[args.spectrum], n_labels=args.n_labels,use_bn=True, basic_out_channel=16, depth=2, dropout=0)
    checkpoint = torch.load("model.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    img = dataset[args.sample]["image"].unsqueeze(0)
    seg = dataset[args.sample]["segmentation"].unsqueeze(0)
    palette = np.array(MUSIC_2D_PALETTE)
    colored_seg = seg.argmax(dim=1).squeeze(0).detach().cpu().numpy()
    colored_seg = palette[colored_seg].astype(np.uint8)

    with torch.no_grad():
        x = model(img)
        pred = x.argmax(dim=1).squeeze(0).detach().cpu().numpy()
        colored_image = palette[pred]
        colored_image = colored_image.astype(np.uint8)
        # image_from_segmentation(x, 16, MUSIC_2D_PALETTE, device="cpu")
        print(mIoU_score(x.argmax(1), seg.argmax(1)).item())
    plt.figure(figsize=(10, 5))

    # Plot seg
    plt.subplot(1, 2, 1)
    plt.title("Seg")
    plt.imshow(colored_seg)
    plt.axis('off')

    # Plot dataset
    plt.subplot(1, 2, 2)
    plt.title("img")
    plt.imshow(colored_image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-dr", "--data_root", type=str, default="./", help="Data root directory")
    parser.add_argument("-e", "--epochs", type=int, default=3000, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.00001, help="Learning rate")
    parser.add_argument("-sp", "--spectrum", type=str, default="reducedSpectrum", help="NUmber of slices")
    parser.add_argument("-s", "--sample", type=int, default=15, help="NUmber of slices")
    parser.add_argument("-n", "--normalize_data", type=bool, default=False, help="Loss function")
    args = parser.parse_args()
    main(args)