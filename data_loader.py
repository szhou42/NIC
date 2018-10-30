# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""


import os
from PIL import Image
from utils import tokenize_captions
from pycocotools.coco import COCO

import torch


class MSCOCO(torch.utils.data.Dataset):

    def __init__(self, transform, val_caption_file, image_dir):
        self.image_dir = image_dir
        
        self.coco = COCO(val_caption_file)
        
        captions = self.coco.anns
        # need to tokenize all captions here
        self.captions = tokenize_captions(captions)
        self.caption_ids = list(captions.keys())

        self.transform = transform

    def __getitem__(self, index):  
        caption_id = self.caption_ids[index]
        caption_and_imageid = self.captions[caption_id]
        
        caption = caption_and_imageid['caption']
        image_id = caption_and_imageid['image_id']
        image_path = os.path.join(self.image_dir, self.coco.loadImgs(image_id)[0]['file_name'])

        image = Image.open(image_path)
        image = self.transform(image)

        return image, caption
    
    def __len__(self):
        return len(self.caption_ids)
