# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""


import pickle
from PIL import Image

import torch


class MSCOCO(torch.utils.data.Dataset):

    def __init__(self, imagepaths_and_captions, transform):
        
        self.imagepaths_captions = pickle.load(open(imagepaths_and_captions, 'rb'))
        
        self.caption_ids = list(self.imagepaths_captions.keys())

        self.transform = transform

    def __getitem__(self, index):  
        caption_id = self.caption_ids[index]
        
        imagepath_and_caption = self.imagepaths_captions[caption_id]
        image_path = imagepath_and_caption['image_path']
        caption = imagepath_and_caption['caption']

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, caption
    
    def __len__(self):
        return len(self.caption_ids)
