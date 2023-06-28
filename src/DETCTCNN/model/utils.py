import PIL
import os
import numpy as np
import seaborn as sns
from PIL import Image

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def image_from_segmentation(prediction,no_classes):
	# Saves the image, the model output and the results after the post processing
    palette = np.rint(np.array(sns.color_palette('hls', no_classes)) * 255)
    mask = prediction.argmax(1).numpy().squeeze()
    colored_image = palette[mask]
    colored_image = colored_image.astype(np.uint8)
    to_save = colored_image.reshape(mask.shape[0], mask.shape[1], 3)
    im = Image.fromarray(to_save)
    im.save("example.jpeg")
    colored_image = colored_image.reshape(3, mask.shape[0], mask.shape[1])
    return colored_image