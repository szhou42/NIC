# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from models import CNN, RNN


DEBUG = True
#DEBUG = False

BATCH_SIZE = 128
EPOCHS = 500
current_epoch = 1
LR = 0.0001
momentum = 0.9 # if SGD

transform = transforms.Compose([])

trainloader = torch.utils.data.DataLoader()
valloader = torch.utils.data.DataLoader()

encoder = CNN()
decoder = RNN()
encoder.cuda()
decoder.cuda()
model_paras = list(encoder.parameters()) + list(RNN.parameters())

#optimizer = optim.SGD(model_paras, lr=LR, momentum=momentum)
optimizer = optim.Adam(model_paras, lr=LR)
criterion = nn.CrossEntropyLoss()

start_time = time.time()
for epoch in range(current_epoch, EPOCHS+1):
    
    encoder.train()
    decoder.train()
    
    for batch_idx, (images, captions) in enumerate(trainloader, 1):
        
        images = images.cuda()
        captions = captions.cuda()
        
        encoder.zero_grad()
        decoder.zero_grad()
        
        image_embeddings = encoder(images)
        generated_captions = decoder(image_embeddings, captions)
        
        loss = criterion(generated_captions, captions)
        
        
        
        
        
