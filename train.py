# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import os
import time
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torchvision import transforms

from models import CNN, RNN
from utils import save_model, load_model
from dataloader import MSCOCO, collate_fn


#DEBUG = True
DEBUG = False
NUM_WORKERS = 0
NO_WORD_EMBEDDINGS = 300
VOCAB_SIZE = 17000 + 3
HIDDEN_SIZE = 512
BATCH_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 200
LR = 2
LR_DECAY_RATE = 0.5
NUM_EPOCHS_PER_DECAY = 4

current_epoch = 1
batch_step_count = 1
time_used_global = 0.0
checkpoint = 1

model_dir = '../saved_model/'
log_dir = '../logs/'
train_imagepaths_and_captions = '../preprocessed_data/imagepaths_captions.train'
val_imagepaths_and_captions = '../preprocessed_data/imagepaths_captions.val'
pretrained_resnet101_file = '../pre_trained/resnet101-5d3b4d8f.pth'
pretrained_word_embeddings_file = '../preprocessed_data/embeddings'
writer = SummaryWriter(log_dir)

transform_train = transforms.Compose([
    transforms.Resize((259, 259)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4731, 0.4467, 0.4059], [0.2681, 0.2627, 0.2774])
])

transform_val = transforms.Compose([
    transforms.Resize((259, 259)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4731, 0.4467, 0.4059], [0.2681, 0.2627, 0.2774])
])

print('Loading dataset...')
trainset = MSCOCO(train_imagepaths_and_captions, transform_train)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                                          shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

valset = MSCOCO(val_imagepaths_and_captions, transform_val)
valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                                        shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

print('Initializing models...')
encoder = CNN(NO_WORD_EMBEDDINGS, pretrained_resnet101_file, freeze=True)
decoder = RNN(VOCAB_SIZE, NO_WORD_EMBEDDINGS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
              pre_trained_file=pretrained_word_embeddings_file, freeze=False)
encoder.cuda()
decoder.cuda()

model_paras = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.SGD(model_paras, lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)


# load lastest model to resume training
model_list = os.listdir(model_dir)
if model_list:
    print('Loading model...')
    state = load_model(model_dir, model_list)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    optimizer.load_state_dict(state['optimizer'])
    current_epoch = state['epoch'] + 1
    time_used_global = state['time_used_global']
    batch_step_count = state['batch_step_count']    

criterion = nn.CrossEntropyLoss()

for epoch in range(current_epoch, EPOCHS+1):
    start_time_epoch = time.time()
    encoder.train()
    decoder.train()

    if epoch % NUM_EPOCHS_PER_DECAY == 0:
        print('Changing learning rate!')
        scheduler.step(int(current_epoch/NUM_EPOCHS_PER_DECAY))

    print('[%d] epoch starts training...'%epoch)
    trainloss = 0.0
    for batch_idx, (images, captions, lengths) in enumerate(trainloader, 1):

        images = images.cuda()
        captions = captions.cuda()
        lengths = lengths.cuda()
        # when doing forward propagation, we do not input end word key; when calculating loss, we do not count start word key.
        lengths -= 1
        # throw out the start word key when calculating loss.
        targets = rnn_utils.pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
        
        encoder.zero_grad()
        decoder.zero_grad()
        
        image_embeddings = encoder(images)
        # throw out the end word key when doing forward propagation.
        generated_captions = decoder(image_embeddings, captions[:, :-1], lengths)

        loss = criterion(generated_captions, targets)
        trainloss += loss

        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            writer.add_scalar('batch/training_loss', loss, batch_step_count)
            batch_step_count += 1
            print('[%d] epoch, [%d] batch, [%.4f] loss, [%.2f] min used.'
                  %(epoch, batch_idx, loss, (time.time()-start_time_epoch)/60))

        if DEBUG:
            break
    trainloss /= batch_idx

    print('[%d] epoch starts validating...'%epoch)
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        valloss = 0.0
        for batch_idx, (images, captions, lengths) in enumerate(valloader, 1):

            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
            lengths -= 1
            targets = rnn_utils.pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

            image_embeddings = encoder(images)
            generated_captions = decoder(image_embeddings, captions[:, :-1], lengths)
    
            loss = criterion(generated_captions, targets)
    
            valloss += loss
    
            if DEBUG:
                break
        valloss /= batch_idx


    time_used_epoch = time.time() - start_time_epoch
    time_used_global += time_used_epoch


    writer.add_scalar('epoch/training_loss', trainloss, epoch)
    writer.add_scalar('epoch/validation_loss', valloss, epoch)
    print('[%d] epoch has finished. [%.4f] training loss, [%.4f] validation loss, [%.2f] min used this epoch, [%.2f] hours used in total'
          %(epoch, trainloss, valloss, time_used_epoch/60, time_used_global/3600))

    if epoch % checkpoint == 0:
        print('Saving model!')
        save_model(model_dir, epoch, batch_step_count, time_used_global, optimizer, encoder, decoder)
        
    if DEBUG:
        break

writer.close()
