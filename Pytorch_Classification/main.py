import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

from pathlib import Path
from sklearn.model_selection import train_test_split

from visualize import *

#####################
# Environment Setup #
#####################

print(f"PyTorch version: {torch.__version__}\n"
      f"torchvision version: {torchvision.__version__}")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)

#####################
#    Data Setup     #
#####################

data_path = Path("./data/KaggleDB_Structured")

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


# plot_transformed_images(list(data_path.glob("*/*.jpg")),
#                         transform=data_transform, 
#                         n=3)

# Load the dataset
dataset = datasets.ImageFolder(root=data_path, transform=data_transform)
classes = dataset.classes
class_dict = dataset.class_to_idx
print(class_dict)

# Convert the dataset to a list of samples
samples = list(dataset)
# Split the samples into training and testing sets
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

# Now you can create DataLoaders from these samples
train_loader = torch.utils.data.DataLoader(dataset=train_samples,
                                           batch_size=32,
                                           num_workers=1,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_samples,
                                          batch_size=32,
                                          num_workers=1,
                                          shuffle=False)

print(train_loader, test_loader)