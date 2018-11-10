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
        x = x.view(x.size(0), -1)
        x = self.fc_output(x)

        return x


class RNN(nn.Module):

    def __init__(self, vocab_size, no_word_embeddings, hidden_size, num_layers, pre_trained_file, freeze):
        super(RNN, self).__init__()
        
        self.id2word = np.array(pickle.load(open('../preprocessed_data/idx2word', 'rb')))
        
        if pre_trained_file is not None:
            pretrained_word_embeddings = torch.from_numpy(pickle.load(open(pre_trained_file, 'rb')).astype(np.float32)).cuda()
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_word_embeddings, freeze)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, no_word_embeddings)

        self.lstm = nn.LSTM(
            input_size=no_word_embeddings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc_output = nn.Linear(hidden_size, vocab_size)

    def decode_idx2word(self, idx_seq):
        return self.id2word[idx_seq]
    
    def forward(self, image_embeddings, captions, lengths):

        word_embeddings = self.word_embeddings(captions)
        
        _, (h_0, c_0) = self.lstm(image_embeddings.view(image_embeddings.size(0), 1, -1))
        
        inputs = rnn_utils.pack_padded_sequence(word_embeddings, lengths, batch_first=True)
        h_t, (h_n, c_n) = self.lstm(inputs, (h_0, c_0))
        
        output = self.fc_output(h_t[0])
        return output

    def greedy_generator(self, image_embeddings, max_caption_length=20, STKidx=1, EDKidx=2):
        batch_size = image_embeddings.size(0)
        _, (h_0, c_0) = self.lstm(image_embeddings.view(batch_size, 1, -1)) # h_0.shape: 1 x batch_size x hidden_size
        output_captions_one_batch = []
        for image_idx in range(batch_size):

            h_image_idx = h_0[:, [image_idx], :]
            c_image_idx = c_0[:, [image_idx], :]
            word_embeddings_image_idx = self.word_embeddings(torch.tensor(STKidx).cuda()).view(1, 1, -1)
            
            output_caption_one_image = []            
            for seq_idx in range(max_caption_length):

                _, (h_image_idx, c_image_idx) = self.lstm(word_embeddings_image_idx, (h_image_idx, c_image_idx))
                output_seq_idx = self.fc_output(h_image_idx)

                predicted_word_idx = torch.argmax(output_seq_idx).item()
                if predicted_word_idx == EDKidx:
                    break
                else:
                    output_caption_one_image.append(predicted_word_idx)
                    word_embeddings_image_idx = self.word_embeddings(torch.tensor(predicted_word_idx).cuda()).view(1, 1, -1)
            
            output_caption_one_image = list(self.decode_idx2word(output_caption_one_image))
            output_captions_one_batch.append(output_caption_one_image)

        return output_captions_one_batch

    def beam_search_generator(self, image_embeddings, max_caption_length=20, STKidx=1, EDKidx=2):
        output_captions_one_batch = None
        return output_captions_one_batch
