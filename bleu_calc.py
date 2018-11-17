# -*- coding: utf-8 -*-
"""
Created in Oct 2018

This program takes two argument: 1 the model name 2 which bleu measure(can take value of 1,2,3, or 4, corresponding to BLEU-1, BLEU-2, BLEU-3, BLEU-4)

Will load the model and evaluate the model's accuracy using the BLEU measure specified

"""

import os
import sys
import time
import pickle
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from models import CNN, RNN
from utils import save_model, load_model, load_model_by_filename
from dataloader import MSCOCO, MSCOCO_VAL, collate_fn, collate_fn_val

MODEL_NAME = sys.argv[1]
WHICH_BLEU = sys.argv[2]
NUM_WORKERS = 0

print(MODEL_NAME, 'starts running!')

init_params_file = '../model_params/' + MODEL_NAME
if os.path.isfile(init_params_file):
    # if resume training, load hypermeters.
    print('Loading params...')
    params = pickle.load(open(init_params_file, 'rb'))

    LR = params['LR']
    VOCAB_SIZE = params['VOCAB_SIZE']
    NO_WORD_EMBEDDINGS = params['NO_WORD_EMBEDDINGS']
    HIDDEN_SIZE = params['HIDDEN_SIZE']
    BATCH_SIZE = params['BATCH_SIZE']
    NUM_LAYERS = params['NUM_LAYERS']

    train_imagepaths_and_captions = params['train_imagepaths_and_captions']
    val_imagepaths_and_captions = params['val_imagepaths_and_captions']
    pretrained_cnn_file = params['pretrained_cnn_file']
    pretrained_word_embeddings_file = params['pretrained_word_embeddings_file']

    transform_train = params['transform_train']
    transform_val = params['transform_val']

    print('Loading models...')
    encoder = params['encoder']
    decoder = params['decoder']
    encoder.cuda()
    decoder.cuda()

    print('Loading optimizer...')
    optimizer = params['optimizer']
    ADAM_FLAG = params['ADAM_FLAG']

else:

   # if tune a new set of hyperparameters or new models, change parameters below before training.
    print('Initilizing params...')
    LR = 4e-4
    VOCAB_SIZE = 17000 + 3
    NO_WORD_EMBEDDINGS = 512
    HIDDEN_SIZE = 512
    BATCH_SIZE = 128
    NUM_LAYERS = 1

    train_imagepaths_and_captions = '../preprocessed_data/imagepaths_captions.train'
    val_imagepaths_and_captions = '../preprocessed_data/imagepaths_captions.val'
    pretrained_cnn_file = '../pre_trained/resnet101-5d3b4d8f.pth'
#    pretrained_word_embeddings_file = '../preprocessed_data/embeddings'
    pretrained_word_embeddings_file = None


    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4731, 0.4467, 0.4059], [0.2681, 0.2627, 0.2774])
    ])

    params = {'LR': LR, 'VOCAB_SIZE': VOCAB_SIZE, 'NO_WORD_EMBEDDINGS': NO_WORD_EMBEDDINGS, 'HIDDEN_SIZE': HIDDEN_SIZE,
              'BATCH_SIZE': BATCH_SIZE, 'NUM_LAYERS': NUM_LAYERS, 'train_imagepaths_and_captions': train_imagepaths_and_captions,
              'val_imagepaths_and_captions': val_imagepaths_and_captions, 'pretrained_cnn_file': pretrained_cnn_file,
              'pretrained_word_embeddings_file': pretrained_word_embeddings_file, 'transform_train': transform_train,
              'transform_val': transform_val}


    print('Initializing models...')
    encoder = CNN(NO_WORD_EMBEDDINGS, pretrained_cnn_file, freeze=True)
    decoder = RNN(VOCAB_SIZE, NO_WORD_EMBEDDINGS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                  pre_trained_file=pretrained_word_embeddings_file, freeze=False)
    params['encoder'] = encoder
    params['decoder'] = decoder
    encoder.cuda()
    decoder.cuda()

    print('Initializing optimizer...')
    model_paras = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(model_paras, lr=LR)
    ADAM_FLAG = True
    params['optimizer'] = optimizer
    params['ADAM_FLAG'] = ADAM_FLAG


model_dir = '../saved_model/'
if os.path.isfile(model_dir + MODEL_NAME):
    state = load_model_by_filename(model_dir, MODEL_NAME)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    optimizer.load_state_dict(state['optimizer'])
else:
    print('Please specify an existing model')
    quit()


print('Loading dataset...')
valset = MSCOCO_VAL(VOCAB_SIZE, '../preprocessed_data/imagepaths_captions.newval', transform_val)
valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=BATCH_SIZE, collate_fn = collate_fn_val, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

print('Starts calculating BLEU-' + str(WHICH_BLEU) + ' score for the val sett.') 
with torch.no_grad():
    encoder.eval()
    decoder.eval()
    bleu_sum = 0.0
    num_val_img = 0
    for batch_idx, (images, captions) in enumerate(valloader, 1):
        images = Variable(images)
        images = images.cuda()
        print('captions: ', captions)
        image_embeddings = encoder(images)
        generated_captions = beam_search_generator(image_embeddings)

        # Calculate bleu score for each of the generated captions
        for i in range(len(generated_captions)):
            bleu = bleu_score(bleu_type, generated_captions[i], captions[i])
            bleu_sum = bleu_sum + bleu
            num_val_img = num_val_img + 1
    bleu_avg = bleu_sum / num_val_img
    print('BLEU Score: ' + str(bleu_avg))
