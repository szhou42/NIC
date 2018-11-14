# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import pickle
import numpy as np
import heapq
import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.rnn as rnn_utils
import math
import pdb

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

        self.softmax = nn.Softmax()

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
            # Get h0 and c0 of some image in the batch
            h_image_idx = h_0[:, [image_idx], :]
            c_image_idx = c_0[:, [image_idx], :]

            # Get Xi = X0 (i.e word embeddings of the start word)
            word_embeddings_image_idx = self.word_embeddings(torch.tensor(STKidx).cuda()).view(1, 1, -1)
            
            output_caption_one_image = []            
            for seq_idx in range(max_caption_length):
                # Given last hidden state, cell state and Xi, obtain the next hidden state and cell state
                _, (h_image_idx, c_image_idx) = self.lstm(word_embeddings_image_idx, (h_image_idx, c_image_idx))
                # Output a vector of len = vocab_size 
                output_seq_idx = self.fc_output(h_image_idx)
                # Take the word index with biggest value
                predicted_word_idx = torch.argmax(output_seq_idx).item()
                if predicted_word_idx == EDKidx:
                    break
                else:
                    output_caption_one_image.append(predicted_word_idx)
                    # Update next Xi to be the word embeddings of the predicted word
                    word_embeddings_image_idx = self.word_embeddings(torch.tensor(predicted_word_idx).cuda()).view(1, 1, -1)
            
            # Convert word indices to actual words 
            output_caption_one_image = list(self.decode_idx2word(output_caption_one_image))
            output_captions_one_batch.append(output_caption_one_image)

        return output_captions_one_batch

    # don't use this yet.. has bug
    def beam_search_generator(self, image_embeddings, max_caption_length=20, STKidx=1, EDKidx=2):
        # Initially, there is only an empty sequence
        beam_width = 1 

        batch_size = image_embeddings.size(0)
        _, (h_0, c_0) = self.lstm(image_embeddings.view(batch_size, 1, -1)) # h_0.shape: 1 x batch_size x hidden_size
        sequences = [[[STKidx], 0.0, h_0, c_0]]
        output_captions_one_batch = []
        #pdb.set_trace()
        for image_idx in range(batch_size):
            # Get h0 and c0 of some image in the batch
            h_image_idx = h_0[:, [image_idx], :]
            c_image_idx = c_0[:, [image_idx], :]

            output_caption_one_image = []
            all_candidates = []
            for seq_idx in range(max_caption_length):
                print('seq_idx = ', seq_idx);
                for k in range(len(sequences)):
                    seq, score, h_curr, c_curr = sequences[k]
                    last_word_in_sequence = seq[len(seq) - 1]
                    strseq = self.decode_idx2word(seq)
                    print('the seq: ', strseq)

                    # This sequence is finished, no need to generate more words for it(but we still want it in the candidate list)
                    if last_word_in_sequence == EDKidx:
                        all_candidates.append(sequences[k])
                        continue

                    word_embeddings_image_idx = self.word_embeddings(torch.tensor(last_word_in_sequence).cuda()).view(1, 1, -1)
                    # Given last hidden state, cell state and Xi, obtain the next hidden state and cell state
                    _, (h_image_idx, c_image_idx) = self.lstm(word_embeddings_image_idx, (h_curr, c_curr))
                    # Output a vector of len = vocab_size 
                    output_seq_idx = self.fc_output(h_image_idx)
                    # Take top k words's indices with biggest values
                    output_seq_idx = output_seq_idx.resize(17003)
                    softmax_output = self.softmax(output_seq_idx)
                    #softmax_output = output_seq_idx

                    topk_results = softmax_output.topk(beam_width, largest=True)
                    topk_indices = topk_results[1]
                    #if strseq[len(strseq) - 1] == 'and':
                    #    print(softmax_output)
                    print('top k words:');
                    for j in range(topk_indices.size()[0]):
                        index = topk_indices[j]
                        candidate = [seq + [index], score + softmax_output[index], h_image_idx, c_image_idx] 
                        all_candidates.append(candidate)
                        print('k = :', j+1, ' ', self.decode_idx2word([index]));

                ordered = sorted(all_candidates, reverse = True, key=lambda tup:tup[1])
                sequences = ordered[:beam_width]

            for p in range(len(sequences)):
                caption = sequences[p][0]
                caption = list(self.decode_idx2word(caption))
                print(caption)

            output_caption_one_image = sequences[0][0]
            # Convert word indices to actual words 
            print(output_caption_one_image)
            output_caption_one_image = list(self.decode_idx2word(output_caption_one_image))
            print(output_caption_one_image)
            output_captions_one_batch.append(output_caption_one_image)

        return output_captions_one_batch
