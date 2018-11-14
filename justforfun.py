'''
Takes a picture(path) and produce a sentence for it
'''

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torchvision import transforms
from models import CNN, RNN
from utils import save_model, load_model
from dataloader import MSCOCO, collate_fn
from PIL import Image


NUM_WORKERS = 8
NO_WORD_EMBEDDINGS = 512
VOCAB_SIZE = 17000 + 3
HIDDEN_SIZE = 512
BATCH_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 200
LR = 2
LR_DECAY_RATE = 0.5
NUM_EPOCHS_PER_DECAY = 4

pretrained_resnet101_file = '../pre_trained/resnet101-5d3b4d8f.pth'
pretrained_word_embeddings_file = None
print("Image path: " + sys.argv[1])
image_path = sys.argv[1]
img = Image.open(image_path)
img = img.convert('RGB')

print(img.size)

transform = transforms.Compose([
    transforms.Resize((259, 259)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4731, 0.4467, 0.4059], [0.2681, 0.2627, 0.2774])
])
img_transformed = transform(img)
print(img_transformed.size())
img_dim = img_transformed.size()
img_transformed = img_transformed.resize_(1, img_dim[0], img_dim[1], img_dim[2])
print(img_transformed.size())


model_dir = '../saved_model/'
print('Initializing models...')
encoder = CNN(NO_WORD_EMBEDDINGS, pretrained_resnet101_file, freeze=True)
decoder = RNN(VOCAB_SIZE, NO_WORD_EMBEDDINGS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
              pre_trained_file=pretrained_word_embeddings_file, freeze=False)
encoder.cuda()
decoder.cuda()

# load lastest model to resume training
model_list = ['model_1.pth'] 
if model_list:
    print('Loading model...')
    state = load_model(model_dir, [model_list[0]])
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        images = img_transformed.cuda()
        image_embeddings = encoder(images)
        #generated_captions = decoder.greedy_generator(image_embeddings)
        generated_captions = decoder.beam_search_generator(image_embeddings)
        print(generated_captions)
