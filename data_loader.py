# -*- coding: utf-8 -*-
"""
Created in Oct 2018

@author: msq96
"""


import os
import torch
import itertools
from PIL import Image


image_dir = '../val2017'
images = os.listdir(image_dir)
image_paths = list(map(os.path.join, itertools.repeat(image_dir), images))




class MSCOCO(torch.utils.data.Dataset):
    def __init__(self):
        pass
        
    def __getitem__(self, index):      
        image = None
        caption = None
        
        return image, caption
    
    def __len__(self):
        return len()
