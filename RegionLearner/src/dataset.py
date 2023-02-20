import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.io import read_image


class RegionDataset(Dataset):
    """Object representing the RegionDataset, that extends from torch.utils.data.Dataset."""  
      
    def __init__(self, x, y, total_regions, height=224, width=224, device="cuda", 
                 use_augmentation=False, brightness=None, contrast=None, saturation=None, hue=None,
                 random_perspective_distortion=None, random_perspective_p=None, random_rotation_degrees=None):
        """Constructor for RegionDataset.

        Args:
            x (list(str)): the paths to the images.
            y (list(int)): the corresponding regions of membership.
            total_regions (int): the number of total regions.
            height (int, optional): the height of the images. Defaults to 224.
            width (int, optional): the width of the images. Defaults to 224.
            device (str, optional): the device into which to move the dataset. Defaults to "cuda".
            use_augmentation (bool, optional): whether use data augmentation. Defaults to False.
            brightness (tuple(float), optional): the brightness range for data augmentation. Defaults to None.
            contrast (tuple(float), optional): the contrast range for data augmentation. Defaults to None.
            saturation (tuple(float), optional): the saturation range for data augmentation. Defaults to None.
            hue (tuple, optional): the hue range for data augmentation. Defaults to None.
            random_perspective_distortion (float, optional): the degree of distortion for data augmentation. Defaults to None.
            random_perspective_p (float, optional): the probability of distortion for data augmentation. Defaults to None.
            random_rotation_degrees (tuple(float), optional): the range of rotation for data augmentation. Defaults to None.
        """        
        self.x = x
        self.y = y
        self.total_regions = total_regions
        self.height = height
        self.width = width
        self.device = device
        self.use_augmentation = use_augmentation
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_perspective_distortion = random_perspective_distortion
        self.random_perspective_p = random_perspective_p
        self.random_rotation_degrees = random_rotation_degrees
        
        if self.use_augmentation:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue),
                torchvision.transforms.RandomPerspective(self.random_perspective_distortion, self.random_perspective_p),
                torchvision.transforms.RandomRotation(self.random_rotation_degrees, torchvision.transforms.InterpolationMode.BILINEAR)
            ])
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        # one hot labels for zones probabilities.
        y = self.y[idx]
        label = np.zeros(self.total_regions) 
        label[y] = 1
        label = torch.tensor(label)
        
        image = torchvision.transforms.Resize((self.height, self.width))(read_image(self.x[idx])) / 255
        
        if self.use_augmentation:
            image = self.transform(image)
        
        return image.to(self.device), label.to(self.device)