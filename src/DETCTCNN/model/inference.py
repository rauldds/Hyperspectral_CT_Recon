from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
from src.DETCTCNN.model.model import get_model
import torch
from src.DETCTCNN.data.music_2d_labels import MUSIC_2D_LABELS, MUSIC_2D_PALETTE
from  src.DETCTCNN.data.music_2d_dataset import MUSIC2DDataset
LABELS_SIZE = len(MUSIC_2D_LABELS)
INPUT_CHANNELS ={
    "reducedSpectrum": 10,
    "fullSpectrum":128
}

def main(args):
    dataset = MUSIC2DDataset(path2d=args.data_root, path3d=None, 
                            spectrum=args.spectrum, partition="valid",
                            full_dataset=False)
    model = get_model(input_channels=INPUT_CHANNELS [args.spectrum], n_labels=args.n_labels,use_bn=True)
    checkpoint = torch.load("./src/DETCTCNN/model/model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    dataset = torch.from_numpy(np.asarray(dataset[2]["image"])).view((1,10,100,100))
    
    palette = np.array(MUSIC_2D_PALETTE)

    with torch.no_grad():
        x = model(dataset)
        pred = x.argmax(dim=1).squeeze(0).detach().cpu().numpy()
        colored_image = palette[pred]
        colored_image = colored_image.astype(np.uint8)
        #print(colored_image)
    plt.figure()
    plt.title("Prediction")
    plt.imshow(colored_image)
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-dr", "--data_root", type=str, default="../../../MUSIC2D_HDF5", help="Data root directory")
    parser.add_argument("-e", "--epochs", type=int, default=700, help="Number of maximum training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-nl", "--n_labels", type=int, default=LABELS_SIZE, help="Number of labels for final layer")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.00005, help="Learning rate")
    parser.add_argument("-sp", "--spectrum", type=str, default="reducedSpectrum", help="NUmber of slices")
    args = parser.parse_args()
    main(args)