# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from models import CNN, RNN
from utils import save_model, load_model
from data_loader import MSCOCO


DEBUG = True
#DEBUG = False

NO_WORD_EMBEDDINGS = 300
BATCH_SIZE = 128
EPOCHS = 500
LR = 0.0001
#MOMENTUM = 0.9 # if SGD
current_epoch = 1
time_used_global = 0.0
model_dir = '../saved_model/'
image_dir = '../data/'
pre_train_dir = '../pre_train/'
checkpoint = 5


trainloader = torch.utils.data.DataLoader()
valloader = torch.utils.data.DataLoader()

encoder = CNN()
decoder = RNN()
encoder.cuda()
decoder.cuda()

model_paras = list(encoder.parameters()) + list(RNN.parameters())
#optimizer = optim.SGD(model_paras, lr=LR, momentum=MOMENTUM)
optimizer = optim.Adam(model_paras, lr=LR)


# load lastest model to resume training
model_list = os.listdir(model_dir)
if model_list:
    state = load_model(model_dir, model_list)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    optimizer.load_state_dict(state['optimizer'])
    current_epoch = state['epoch'] + 1
    time_used_global = state['time_used_global']

criterion = nn.CrossEntropyLoss()

for epoch in range(current_epoch, EPOCHS+1):
    start_time_epoch = time.time()
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
        
    time_used_epoch = time.time() - start_time_epoch
    time_used_global += time_used_epoch
    
    if epoch % checkpoint == 0:
        save_model(epoch, time_used_global, optimizer, encoder, decoder)

        if DEBUG:
            break

    if DEBUG:
        break        
