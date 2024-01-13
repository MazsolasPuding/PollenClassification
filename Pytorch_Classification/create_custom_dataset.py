"""
This module is for loading the Pollen Dataset,
and creating a custom dataset sublcassing the PyTorch Dataset class.
"""

import os
import torch

from pprint import pprint
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List, Set
from pathlib import Path

class PollenDataset(Dataset):
    """Pollen dataset."""

    def __init__(self, 
                 root_dir: Path,
                 transform: transforms.Compose = None) -> None:
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (torchvision.transforms.Compose, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._get_samples()

    def _get_classes(self) -> Set[str]:
        pictures = [item.name for item in self.root_dir.iterdir()]  # Some pictures have spaces instead of underscores
        return list(set([pic.replace(' ', '_').split('_')[0] for pic in pictures]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.samples[idx][0]
        image = Image.open(img_name)
        label = self.samples[idx][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_samples(self) -> List[Tuple[str, int]]:
        """Creates a list of samples from the dataset.

        Returns:
            List[Tuple[str, int]]: List of samples from the dataset.
        """
        samples = []
        for pic in self.root_dir.iterdir():
            cls = pic.name.replace(' ', '_').split('_')[0]
            samples.append([pic, self.class_to_idx[cls]])
        return samples

if __name__ == "__main__":
    p = PollenDataset(root_dir="./data/KaggleDB")
    pprint(Image.open(p.samples[0][0]))