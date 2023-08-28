import PIL
import os
import numpy as np
import seaborn as sns
from PIL import Image
from numpy import inf
import torch
from sklearn.utils.class_weight import compute_class_weight

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def image_from_segmentation(prediction,no_classes, palette, device):
    for i in range(prediction.shape[0]):
        cur_pred = prediction[i].unsqueeze(0)
        palette = np.array(palette)
	    # Saves the image, the model output and the results after the post processing
        if device == 'cuda':
            cur_pred = cur_pred.detach().cpu()
        mask = cur_pred.detach().cpu().argmax(1).numpy().squeeze()
        colored_image = palette[mask]
        colored_image = colored_image.astype(np.uint8)
        to_save = colored_image.reshape(mask.shape[0], mask.shape[1], 3)
        im = Image.fromarray(to_save)
        im.save(f"example{i}.jpeg")
        colored_image = colored_image.reshape(3, mask.shape[0], mask.shape[1])
        return colored_image

def plot_segmentation(segmentation, palette):
    palette = np.array(palette)
	# Saves the image, the model output and the results after the post processing
    mask = segmentation.argmax(1).numpy().squeeze()
    colored_image = palette[mask]
    colored_image = colored_image.astype(np.uint8)
    to_save = colored_image.reshape(mask.shape[0], mask.shape[1], 3)
    im = Image.fromarray(to_save)
    im.save("segmentation.jpeg")
    colored_image = colored_image.reshape(3, mask.shape[0], mask.shape[1])
    return colored_image


def class_frequencies(dataset, n_classes):
    """ Calculates the loss weights of a dataset"""
    freqs = np.zeros(n_classes)
    for sample in dataset:
        gt = sample["segmentation"].argmax(0)
        ind, count = np.unique(gt, return_counts=True)
        # Add previous freq to current count
        freqs[ind] = freqs[ind] + count
    return freqs

def class_weights(dataset, n_classes):
      freqs = class_frequencies(dataset=dataset, n_classes=n_classes)
      med = np.median(freqs[freqs != 0])
      w_s = med/freqs
      # Infrequent classes have high loss
      w_s[w_s == inf] = 100
      w_s = torch.from_numpy(w_s)
      return w_s/sum(w_s)

def flatten_data(dataset):
    flattened_dataset = np.asarray([0])
    for sample in dataset:
        gt = sample["segmentation"].argmax(0).cpu().detach().numpy()
        flattened_dataset = np.concatenate([flattened_dataset.reshape(1,-1),gt.flatten().reshape(1,-1)], 1)
        #flattened_dataset.stack(gt.flatten())
    print((flattened_dataset).shape)
    return np.asarray(flattened_dataset).flatten()

def class_weights_sklearn(dataset, n_classes):
    print(n_classes)
    flat_dataset = flatten_data(dataset)
    class_weights = compute_class_weight('balanced', classes=range(n_classes), y=flat_dataset)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights

def calculate_data_statistics(dataset):
    stacked_data = torch.stack(dataset,dim=0)
    mean = stacked_data.mean(dim=(0,2,3))
    std = stacked_data.std(dim=(0,2,3))
    return mean, std

def calculate_min_max(dataset):
    stacked_data = torch.stack(dataset,dim=0)
    max = stacked_data.amax(dim=(0,2,3))
    min = stacked_data.amin(dim=(0,2,3)) 
    return min, max

def calculate_min_max2(dataset):
    stacked_data = torch.stack(dataset,dim=0)
    max = stacked_data.max()
    min = stacked_data.min() 
    return min, max

def standardize(tensor, mean, std):
     mean = mean.view(-1,1,1)
     std = std.view(-1,1,1)
     return (tensor - mean)/std

def normalize(tensor, min, max):
     min = min.view(-1,1,1)
     max = max.view(-1,1,1)
     return (tensor - min)/(max-min)