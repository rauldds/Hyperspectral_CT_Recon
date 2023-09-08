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
        self.fcext = torch.nn.Linear(in_features=936, out_features=936)
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
        x4 = self.fcext(x4)
        x5 = self.fc2(x4)

        x5 = x5.view((batch_size, 16, 33))

        x6 = self.dec1(x5)
        x7 = self.dec2(x6)
        x8 = self.dec3(x7)


        x9 = self.convolutions(x8)
        
        out = nn.functional.softmax(x9,dim=2)

        return out


class OneDLogRegSkip(torch.nn.Module):
    def __init__(self):
        super(OneDLogRegSkip, self).__init__()

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
        self.fcext = torch.nn.Linear(in_features=936, out_features=936)
        self.fc2 = torch.nn.Linear(in_features=936, out_features=1872)

        # Decoder
        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=16+16, out_channels=32, kernel_size=4),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6)
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=32+32, out_channels=64, kernel_size=5),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.7)
        )
        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=64+64, out_channels=128, kernel_size=5),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.8)
        )

        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=4, stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv1d(in_channels=32, out_channels=1, kernel_size=2, stride=2),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=2),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=2),
            torch.nn.BatchNorm1d(1),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # print(f"this is the shape of x @ logRegSkip {x.shape}")
        x1 = self.enc1(x)
        # print(f"this is the shape of x1 @ logRegSkip {x1.shape}")
        x2 = self.enc2(x1)
        # print(f"this is the shape of x2 @ logRegSkip {x2.shape}")
        x3 = self.enc3(x2)
        # print(f"this is the shape of x3 @ logRegSkip {x3.shape}")

        x3_flattened = x3.view((batch_size, -1))
        # print(f"this is the shape of x4 @ logRegSkip {x3_flattened.shape}")

        x4 = self.fc1(x3_flattened)
        # print(f"this is the shape of x4 @ logRegSkip {x4.shape}")
        x4 = self.fcext(x4)
        # print(f"this is the shape of x4 extra @ logRegSkip {x4.shape}")
        x5 = self.fc2(x4)
        # print(f"this is the shape of x5 @ logRegSkip {x5.shape}")

        x5 = x5.view((batch_size, 16, 117))
        # print(f"this is the shape of x5 reshaped @ logRegSkip {x5.shape}")

        x6 = self.dec1(torch.cat(tensors=(x5, x3), dim=1))
        # print(f"this is the shape of x6 @ logRegSkip {x6.shape}")
        x7 = self.dec2(torch.cat(tensors=(x6, x2), dim=1))
        # print(f"this is the shape of x7 @ logRegSkip {x7.shape}")
        x8 = self.dec3(torch.cat(tensors=(x7, x1), dim=1))
        # print(f"this is the shape of x8 @ logRegSkip {x8.shape}")

        x9 = self.convolutions(x8)
        # print(f"this is the shape of x9 @ logRegSkip {x9.shape}")

        out = nn.functional.softmax(x9, dim=2)

        return out
