# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from models import CNN, RNN
from utils import save_model, load_model
from data_loader import MSCOCO


DEBUG = True
#DEBUG = False

NO_WORD_EMBEDDINGS = 300
VOCAB_SIZE = 20000
HIDDEN_SIZE = 1000
BATCH_SIZE = 128
NUM_LAYERS = 1
EPOCHS = 500
LR = 0.0001
#MOMENTUM = 0.9 # if SGD
current_epoch = 1
time_used_global = 0.0
checkpoint = 5

model_dir = '../saved_model/'
train_image_dir = '../data/train2017/'
val_image_dir = '../data/val2017/'
train_caption_file = '../data/annotations/captions_train2017.json'
val_caption_file = '../data/annotations/captions_val2017.json'
pretrained_resnet101_file = '../pre_trained/resnet101-5d3b4d8f.pth'
pretrained_word_embeddings_file = '../pre_trained/glove.840B.300d.txt' # need to turn it into pytorch.tensor type beforehand.


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

trainset = MSCOCO(transform_train, train_caption_file, train_image_dir)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

valset = MSCOCO(transform_val, val_caption_file, val_image_dir)
valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

encoder = CNN(NO_WORD_EMBEDDINGS, pretrained_resnet101_file, freeze=True)
decoder = RNN(VOCAB_SIZE, NO_WORD_EMBEDDINGS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
              dropout=0.5, pre_trained_file=pretrained_word_embeddings_file, freeze=True)
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
