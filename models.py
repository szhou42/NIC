# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

    def forward(self, x):
        
        return x


class RNN(nn.Module):
    
    def __init__(self):
        super(RNN, self).__init__()
    
    def forward(self, image_embeddings, captions):
        h = None
        return h