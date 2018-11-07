# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import pickle
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.rnn as rnn_utils

def resnet101(pre_trained_file, pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
    if pretrained:
        state_dict = torch.load(open(pre_trained_file, 'rb'))
        model.load_state_dict(state_dict)
    return model


class CNN(nn.Module):
    
    def __init__(self, batch_size, no_word_embeddings, pre_train_dir, freeze):
        super(CNN, self).__init__()

        pretrained_resnet101 = resnet101(pre_train_dir, pretrained=True)
        self.batch_size = batch_size

        self.resnet = nn.Sequential(*list(pretrained_resnet101.children())[:-1])
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad_(requires_grad=False)

        self.fc_output = nn.Linear(pretrained_resnet101.fc.in_features, no_word_embeddings)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(self.batch_size, -1)
        x = self.fc_output(x)

        return x


class RNN(nn.Module):

    def __init__(self, batch_size, vocab_size, no_word_embeddings, hidden_size, num_layers, pre_trained_file, freeze):
        super(RNN, self).__init__()
        
        self.batch_size = batch_size

        pretrained_word_embeddings = torch.from_numpy(pickle.load(open(pre_trained_file, 'rb')).astype(np.float32)).cuda()
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_word_embeddings, freeze)

        self.lstm = nn.LSTM(
            input_size=no_word_embeddings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc_output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, image_embeddings, captions, lengths):

        word_embeddings = self.word_embeddings(captions)
        
        _, (h_0, c_0) = self.lstm(image_embeddings.view(self.batch_size, 1, -1))
        
        inputs = rnn_utils.pack_padded_sequence(word_embeddings, lengths, batch_first=True)
        h_t, (h_n, c_n) = self.lstm(inputs, (h_0, c_0))
        
        output = self.fc_output(h_t[0])
        return output




#        all_embeddings = torch.cat((image_embeddings.view(self.batch_size, 1, -1), word_embeddings), dim=1)
#        inputs = rnn_utils.pack_padded_sequence(all_embeddings, lengths, batch_first=True)
#        h_t, (h_n, c_n) = self.lstm(inputs)
