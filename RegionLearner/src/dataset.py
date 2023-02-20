import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.io import read_image


class RegionDataset(Dataset):
    def __init__(self, x, y, total_regions, height=224, width=224, device="cuda", 
                 use_augmentation=False, brightness=None, contrast=None, saturation=None, hue=None,
                 random_perspective_distortion=None, random_perspective_p=None, random_rotation_degrees=None):
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
        y = torch.tensor(self.y[idx])
        label = np.zeros(self.total_regions)
        label[y] = 1
        
        label = torch.tensor(label)
        image = torchvision.transforms.Resize((self.height, self.width))(read_image(self.x[idx])) / 255
        
        if self.use_augmentation:
            image = self.transform(image)
        
        return image.to(self.device), label.to(self.device)