import random
from PIL import Image
import  matplotlib.pyplot as plt

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