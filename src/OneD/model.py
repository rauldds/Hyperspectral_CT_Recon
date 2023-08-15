import torch
from config import hparams


class OneDAutoEncoder(torch.nn.Module):
    def __init__(self, input_channels=128):
        super(OneDAutoEncoder, self).__init__()

        # Encoder
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8)
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.7)
        )
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6)
        )
        # Latent Space
        self.fc1 = torch.nn.Linear(in_features=19984, out_features=8000)
        self.fc3 = torch.nn.Linear(in_features=8000, out_features=19984)

        # Decoder
        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=16+16, out_channels=32, kernel_size=4, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6)
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=32+32, out_channels=64, kernel_size=5, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.7)
        )
        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=64+64, out_channels=128, kernel_size=5, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8)
        )
        self.adjustment = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)

        # Decreasing the channels to match Segmentation shape
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1)
        torch.nn.BatchNorm1d(16)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.enc1(x)
        # print(f"x1 shape: {x1.shape}")
        x2 = self.enc2(x1)
        # print(f"x2 shape: {x2.shape}")
        x3 = self.enc3(x2)
        # print(f"x3 shape: {x3.shape}")

        flattened_x3 = x3.view((batch_size, -1))
        # print(f"flattened x3 shape: {flattened_x3.shape}")

        x4 = self.fc1(flattened_x3)
        # print(f"x4 shape: {x4.shape}")
        x5 = self.fc3(x4)
        # print(f"x5 shape: {x5.shape}")

        x5 = x5.view((batch_size, 16, 1249))
        # print(f"x5 flattened shape: {x5.shape}")

        x6 = self.dec1(torch.cat((x5, x3), 1))  # Concatenate along the channel dimension
        # print(f"x6 shape: {x6.shape}")
        x7 = self.dec2(torch.cat((x6, x2), 1))
        # print(f"x7 shape: {x7.shape}")
        x8 = self.dec3(torch.cat((x7, x1), 1))
        # print(f"x8 shape: {x8.shape}")
        x8 = self.adjustment(x8)

        x9 = self.conv1(x8)
        # print(f"x9 shape: {x9.shape}")
        x10 = self.conv2(x9)
        # print(f"x10  shape: {x10.shape}")
        x11 = self.conv3(x10)
        # print(f"x11  shape: {x11.shape}")

        x11 = x11.view(batch_size, 16, 100, 100)
        # print(f"x11  shape: {x11.shape}")

        return x11
