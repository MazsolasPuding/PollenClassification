import random
import shutil
from typing import List, Dict
from collections import Counter
import os

import torch
import torchinfo
from pathlib import Path
import  matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns

from data_setup import PollenDataset

###################
#  File Handling  #
###################

def convert_db_to_standard_img_cls_format(path: str = "/Users/horvada/Git/PERSONAL/PollenClassification/data/KaggleDB"):
    database_path = Path(path)

    pictures = [item.name for item in database_path.iterdir()]  # Some pictures have spaces instead of underscores
    classes = set([pic.replace(' ', '_').split('_')[0] for pic in pictures])
    
    new_data_path = database_path.parent / "KaggleDB_Structured"
    new_data_path.mkdir(parents=True, exist_ok=True)

    for ind, _class in enumerate(classes):
        new_sub = new_data_path / _class
        new_sub.mkdir(parents=True, exist_ok=True)

        for pic in pictures:
            if pic.replace(' ', '_').split('_')[0] == _class:
                shutil.copy(database_path / pic, (new_sub / str(ind)).with_suffix(".jpg"))


def save_model(model: torch.nn.Module,
                model_name: str,
                target_dir: str = "./models"):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    
def test_data_loader(dataloader= torch.utils.data.DataLoader):
    # Get image and label from custom DataLoader
    img_custom, label_custom = next(iter(dataloader))
    print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label_custom.shape}")
    
def test_inference(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: str = "cpu"):
    
    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(dataloader))

    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # 3. Perform a forward pass on a single image
    model.to(device)
    model.eval()
    with torch.inference_mode():
        pred = model(img_single.to(device))

    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")

def print_model_info(model: torch.nn.Module,
                     info: str = "Basic"):
    # Print Basic info
    if info in ["Basic", "Both"]: print(model)
    # Print Complete Modell Summary
    if info in ["Complete", "Both"]: print(torchinfo.summary(model=model, 
                                                             input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
                                                             # col_names=["input_size"], # uncomment for smaller output
                                                             col_names=["input_size", "output_size", "num_params", "trainable"],
                                                             col_width=20,
                                                             row_settings=["var_names"]))

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

def show_classe_distribution(data_path: str = "./data/KaggleDB"):
    names = [name.replace(' ', '_').split('_')[0] for name in os.listdir(data_path)]
    classes = Counter(names)  #returns dictionary with class names and their counts
    plt.figure(figsize = (12,8))
    plt.title('Class Counts in Dataset')
    plt.bar(*zip(*classes.items()))
    plt.xticks(rotation='vertical')
    plt.show()

def show_scatter_plot(data_path: str = "./data/KaggleDB"):
    """Show scatter plot of image sizes."""
    size = [cv2.imread(os.path.join(data_path, name)).shape for name in os.listdir(data_path)]
    x, y, _ = zip(*size)
    fig = plt.figure(figsize=(12, 10))

    # scatter plot
    plt.scatter(x,y)
    plt.title("Image size scatterplot")

    # add diagonal red line 
    max_dim = max(max(x), max(y))
    plt.plot([0, max_dim],[0, max_dim], 'r')
    plt.show()

def show_images(data_path: str = "./data/KaggleDB"):
    path_class =  {cls: [os.path.join(data_path, name)
                  for name in os.listdir(data_path)
                  if name.replace(' ', '_').split('_')[0] == cls]
                for cls in PollenDataset._get_classes(data_path)}
    
    fig = plt.figure(figsize=(15, 15))
    for i, key in enumerate(path_class.keys()):
        img1 = Image.open(path_class[key][0]) 
        img2 = Image.open(path_class[key][1]) 
        img3 = Image.open(path_class[key][2]) 

        ax = fig.add_subplot(8, 9,  3*i + 1, xticks=[], yticks=[])
        ax.imshow(img1)
        ax.set_title(key)
        
        ax = fig.add_subplot(8, 9,  3*i + 2, xticks=[], yticks=[])
        ax.imshow(img2)
        ax.set_title(key)

        ax = fig.add_subplot(8, 9,  3*i + 3, xticks=[], yticks=[])
        ax.imshow(img3)
        ax.set_title(key)
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

def plot_confusion_matrix(model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          device: str = "cpu"):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    pred_labels = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
