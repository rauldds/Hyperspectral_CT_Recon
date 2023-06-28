import PIL
import os
import numpy as np
import seaborn as sns

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def image_from_segmentation(prediction,no_classes):
	# Saves the image, the model output and the results after the post processing
    palette = sns.color_palette('hls', no_classes)
    mask = prediction.argmax(1).numpy().squeeze()
    colored_image = np.take(palette, mask, axis=0).astype(np.uint8)
    colored_image = colored_image.transpose(2,0,1)
    return colored_image