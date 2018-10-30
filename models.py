# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import torch
import torch.nn as nn
import torchvision

def resnet101(pretrained=True, pre_train_dir):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url('http://download.pytorch.org/models/resnet101-5d3b4d8f.pth', model_dir=pre_train_dir))
    return model


class CNN(nn.Module):
    
    def __init__(self, no_word_embeddings, pre_train_dir, freeze):
        super(CNN, self).__init__()
        pretrained_resnet101 = resnet101(pretrained=True, pre_train_dir)
        self.resnet = nn.Sequential(*list(pretrained_resnet101.children())[:-1])
        if freeze:
            self.resnet.weight.requires_grad = False

        self.fc_output = nn.Linear(pretrained_resnet101.fc.in_features, no_word_embeddings)

    def forward(self, x):
        x = self.resnet(x)        
        x = self.fc_output(x)

        return x


class RNN(nn.Module):
    
    def __init__(self, pre_train_dir, freeze):
        super(RNN, self).__init__()
        pretrained_word_embeddings = torch.load(open(pre_train_dir, 'rb'))
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_word_embeddings, freeze)
    
    def forward(self, image_embeddings, captions):
        h = None
        return h