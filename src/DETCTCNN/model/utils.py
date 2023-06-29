import PIL
import os
import numpy as np
import seaborn as sns
from PIL import Image
from numpy import inf
import torch

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def image_from_segmentation(prediction,no_classes):
	# Saves the image, the model output and the results after the post processing
    palette = np.rint(np.array(sns.color_palette('hls', no_classes-1)) * 255)
    palette = np.insert(palette, 0, [0,0,0], axis=0)
    mask = prediction.argmax(1).numpy().squeeze()
    colored_image = palette[mask]
    colored_image = colored_image.astype(np.uint8)
    to_save = colored_image.reshape(mask.shape[0], mask.shape[1], 3)
    im = Image.fromarray(to_save)
    im.save("example.jpeg")
    colored_image = colored_image.reshape(3, mask.shape[0], mask.shape[1])
    return colored_image


def class_frequencies(dataset, n_classes):
    """ Calculates the loss weights of a dataset"""
    freqs = np.zeros(n_classes)
    for sample in dataset:
        gt = sample["segmentation"].argmax(0)
        ind, count = np.unique(gt, return_counts=True)
        freqs[ind] = count
    return freqs

def class_weights(dataset, n_classes):
      freqs = class_frequencies(dataset=dataset, n_classes=n_classes)
      print(freqs)
      med = np.median(freqs[freqs != 0])
      w_s = med/freqs
      w_s[w_s == inf] =  0
      return torch.from_numpy(w_s)
