import torch

import torch.nn as nn

class OneDLogReg(torch.nn.Module):
    def __init__(self):
        super(OneDLogReg, self).__init__()

        # Encoder
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8)
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.7)
        )
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6)
        )

        # Latent Space
        self.fc1 = torch.nn.Linear(in_features=1872, out_features=936)
        self.fc2 = torch.nn.Linear(in_features=936, out_features=528)

        # Decoder
        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6)
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.7)
        )
        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8)
        )

        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32, out_channels=1, kernel_size=12, stride=2),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=2),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=2),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=2),
            torch.nn.BatchNorm1d(1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x3_flattened = x3.view((batch_size, -1))

        x4 = self.fc1(x3_flattened)
        x5 = self.fc2(x4)

        x5 = x5.view((batch_size, 16, 33))

        x6 = self.dec1(x5)
        x7 = self.dec2(x6)
        x8 = self.dec3(x7)


        x9 = self.convolutions(x8)
        
        out = nn.functional.softmax(x9,dim=2)

        return out
