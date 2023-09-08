import os
import numpy as np

import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict
from PIL import Image
import torch as th


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value/normalize for key, value in self.results.items()}

def image_from_segmentation(prediction,no_classes, palette, device, t=1):
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
        im.save(f"example{t}.jpeg")
        colored_image = colored_image.reshape(3, mask.shape[0], mask.shape[1])
        return colored_image

def patchify_image(I, patch_size):
    spec = I.shape[1]
    return I.unfold(2, patch_size, patch_size).unfold(3,patch_size,patch_size).squeeze().permute((1,2,0,3,4)).reshape(-1,spec, patch_size, patch_size)



