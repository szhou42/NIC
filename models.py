# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import pickle
import torch
import torch.nn as nn
import torchvision

def resnet101(pre_trained_file, pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
    if pretrained:
        state_dict = torch.load(open(pre_trained_file, 'rb'))
        model.load_state_dict(state_dict)
    return model


class CNN(nn.Module):
    
    def __init__(self, no_word_embeddings, pre_train_dir, freeze):
        super(CNN, self).__init__()

        pretrained_resnet101 = resnet101(pre_train_dir, pretrained=True)
        self.resnet = nn.Sequential(*list(pretrained_resnet101.children())[:-1])
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad_(requires_grad=False)

        self.fc_output = nn.Linear(pretrained_resnet101.fc.in_features, no_word_embeddings)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, x.size(1))
        x = self.fc_output(x)

        return x


class RNN(nn.Module):

    def __init__(self, vocab_size, no_word_embeddings, hidden_size, num_layers,
                 dropout, pre_trained_file, freeze):
        super(RNN, self).__init__()

        pretrained_word_embeddings = torch.from_numpy(pickle.load(open(pre_trained_file, 'rb'))).cuda()
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_word_embeddings, freeze)

        self.lstm = nn.LSTM(
            input_size=no_word_embeddings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
#            batch_first=True
        )
        
        self.fc_output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, image_embeddings, captions):
        
        word_embeddings = self.word_embeddings(captions)
        # here need to make the image_embeddings be h_0 and then feed to self.lstm
        h_t, (h_n, c_n) = self.lstm(word_embeddings)

        output = self.fc_output(h_t)
        return output
