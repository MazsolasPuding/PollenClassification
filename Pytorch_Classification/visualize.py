import random
import  matplotlib.pyplot as plt
import torch

from PIL import Image
from typing import List, Dict

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)

    fig, ax = plt.subplots(3, 2)
    for ind, image_path in enumerate(random_image_paths):
        with Image.open(image_path) as f:
            ax[ind][0].imshow(f) 
            ax[ind][0].set_title(f"Original \nSize: {f.size}")
            ax[ind][0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[ind][1].imshow(transformed_image) 
            ax[ind][1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[ind][1].axis("off")

    fig.suptitle(f"Show Transformation step", fontsize=16)
    plt.subplots_adjust(top=0.8, bottom=0.1, hspace=0.6)
    plt.show()

def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          displa_shape:bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        displa_shape = False
    if seed:
        random.seed(seed)

    random_sample_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))

    for i, sample in enumerate(random_sample_idx):
        sample, label = dataset[sample]
        sample_converted = sample.permute(1, 2, 0)

        plt.subplot(1, n, i+1)
        plt.imshow(sample_converted)
        plt.axis("off")
        if classes:
            title = F"Class: {classes[label]}"
            if displa_shape:
                title = title + f"\nshape: {sample_converted.shape}"
            plt.title(title)
    plt.show()

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


