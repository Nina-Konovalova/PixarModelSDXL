import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import os
import numpy as np
from torchvision import transforms



def is_image_file(file_path: str) -> bool:
    """
    Check if the file is an image based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def get_images_from_directory(directory_path: str) -> List[str]:
    """
    Get a list of all image file paths in the directory.

    Args:
        directory_path (str): Path to the directory.

    Returns:
        List[str]: List of image file paths.
    """
    image_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
    return image_files
    

class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        """
        Initialize the ImageDataset.

        Args:
            image_paths (List[str]): List of image file paths.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Image.Image:
        """
        Get the image at the specified index.

        Args:
            idx (int): Index of the image to get.

        Returns:
        
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 
                'name': self.image_paths[idx].split('/')[-1]}


def save_images(images: List[Union[Image.Image, torch.Tensor, np.ndarray]], save_dir: str, img_names: List[str]):
    """
    Save a list of images to the specified directory.

    Args:
        images (List[Union[Image.Image, torch.Tensor, np.ndarray]]): List of images to save.
        save_dir (str): Directory to save the images.
        img_names (List[str]): List of images names
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(os.path.join(save_dir, f'pixar_{img_names[i]}.png'))

