import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime



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

def plot_class_colors_and_accuracies(labels_dict, palette, accuracies, highest_recorded_acc, params):
    fig, ax = plt.subplots(figsize=(10, 10))
    square_size = 0.5

    # Ensure accuracies are in the same order as labels_dict and palette
    sorted_accuracies = [accuracies[cls] for cls in labels_dict.keys()]

    y_pos = np.arange(len(labels_dict))

    # Plot squares for colors
    for idx, color in enumerate(palette):
        ax.add_patch(Rectangle((0, idx), square_size, square_size, facecolor=np.array(color) / 255))

    # Plot text for class names and accuracies
    for idx, (cls, acc) in enumerate(zip(labels_dict.keys(), sorted_accuracies)):
        ax.text(square_size + 0.2, idx + square_size / 2, f"{cls}: {acc:.2f}%", va='center', fontsize=10)

    # Display the highest recorded accuracy and params
    ax.text(square_size + 0.2, len(labels_dict) + 1, f"Highest Recorded Golbal Validation Acc: {100*highest_recorded_acc:.2f}%", fontsize=12)
    ax.text(square_size + 0.2, len(labels_dict) + 2, f"Params: {params}", fontsize=12, wrap=True)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(labels_dict) + 4)  # Added some space for the highest recorded accuracy and params
    ax.axis('off')  # Hide axes

    # Save the image with the current time as a prefix
    current_time = datetime.now().strftime('[%Y.%m.%d_%H:%M.%S]')
    filename = f"experiments_info/{current_time}_class_colors_and_accuracies.png"

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")