import numpy as np
from PIL import Image


def image_from_segmentation(prediction,
                            no_classes,
                            palette,
                            device,
                            mode:str):


    for i in range(prediction.shape[0]):
        cur_pred = prediction[i].unsqueeze(0)
        palette = np.array(palette)
        # Saves the image, the model output and the results after the post-processing
        if device == 'cuda':
            cur_pred = cur_pred.detach().cpu()
        mask = cur_pred.detach().cpu().argmax(1).numpy().squeeze()
        colored_image = palette[mask]
        colored_image = colored_image.astype(np.uint8)
        to_save = colored_image.reshape(mask.shape[0], mask.shape[1], 3)
        im = Image.fromarray(to_save)
        im.save(f"{mode}{i}.jpeg")
        colored_image = colored_image.reshape(3, mask.shape[0], mask.shape[1])
        return colored_image