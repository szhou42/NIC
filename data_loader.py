# -*- coding: utf-8 -*-
"""
Created in Oct 2018

@author: msq96
"""


import os
import itertools
from PIL import Image

import torch
import torchvision.transforms as transforms


image_dir = '../val2017'
images = os.listdir(image_dir)
image_paths = list(map(os.path.join, itertools.repeat(image_dir), images))


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self):
        pass
        
    def __getitem__(self, index):      
        image = None
        caption = None
        
        return image, caption
    
    def __len__(self):
        return len()
