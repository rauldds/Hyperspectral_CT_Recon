import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from src.OneD_CNN.model import OneDAutoEncoder
from src.DETCTCNN.data import MUSIC2DDataset
from src.DETCTCNN.data import MUSIC_2D_LABELS
from src.DETCTCNN.data import MUSIC_2D_PALETTE
from src.OneD_CNN.config import hparams
from utils import image_from_segmentation

LABELS_SIZE = len(MUSIC_2D_LABELS)
log_interval = hparams['log_interval']
initial_epochs_buffer = hparams['initial_epochs_buffer']

tb = SummaryWriter()


# Define the device and allocate our model to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")
model = OneDAutoEncoder()
model.to(device)

# Define optimizer and loss criterion
optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
scheduler  = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
loss_criterion = torch.nn.CrossEntropyLoss().to(device)

epochs = hparams['epochs']

# Setting up the train and validation dataloaders
train_dataset = MUSIC2DDataset(path2d=hparams['dataset_path'], path3d=None, partition="train", spectrum="fullSpectrum",
                               transform=None)
train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
num_batches = len(train_loader)

val_dataset = MUSIC2DDataset(path2d=hparams['dataset_path'], path3d=None, partition="valid", spectrum="fullSpectrum",
                             transform=None)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Setting up early stopping mechanism parameters
best_val_loss = float('inf')
patience = hparams['early_stopping_patience']
no_improvement_epochs = 0
early_stopping_threshold = 1e-4

print(f'[INFO] Training Started')
print(f'[WARNING] Not saving model parameters until epoch {initial_epochs_buffer}\n')
for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        # print(f"images shape: {images.shape}")
        images = images.view((-1, 128, 10000))
        segmentations = batch['segmentation'].to(device)
        optimizer.zero_grad()

        output = model(images)
        loss = loss_criterion(output, segmentations)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            tb.add_scalar("Training Loss", avg_loss, epoch)
            image_from_segmentation(output, LABELS_SIZE, MUSIC_2D_PALETTE, device=device, mode="train")
            print(f"[Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{num_batches}] Training loss: {avg_loss:.3f}")
            running_loss = 0.0  # Reset running loss after logging

    running_val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            images = val_batch['image'].to(device)
            images = images.view((-1, 128, 10000))
            segmentations = val_batch['segmentation'].to(device)

            output = model(images)
            val_loss = loss_criterion(output, segmentations)
            image_from_segmentation(output, LABELS_SIZE, MUSIC_2D_PALETTE, device=device, mode="val")
            running_val_loss += val_loss.item()

    average_val_loss = running_val_loss / len(val_loader)
    # Update the learning rate based on validation loss
    scheduler.step(average_val_loss)
    tb.add_scalar("Validation Loss", average_val_loss, epoch)
    print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {average_val_loss:.4f}")

    # Check for model improvement
    if epoch > initial_epochs_buffer and best_val_loss - average_val_loss > early_stopping_threshold:
        best_val_loss = average_val_loss
        no_improvement_epochs = 0
        print(f"[INFO] New best model parameters saved @ epoch#: {epoch + 1}")
        torch.save(model.state_dict(), 'best_model.pth')
    elif epoch < initial_epochs_buffer:
        no_improvement_epochs = 0

    else:
        no_improvement_epochs += 1
    # print(f"No improvement epochs {no_improvement_epochs}")

    # Check if model should stop early
    if no_improvement_epochs >= patience:
        print(f"Early stopping due to no improvement @ epoch#: {epoch + 1}")
        break
