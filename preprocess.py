# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import pickle
import itertools

import nltk
import numpy as np
from pycocotools.coco import COCO

caption_file = '../data/annotations/captions_train2017.json'
general_caption_info = '../preprocessed_data/general_caption_info'
pretrian_lookup_table = '../preprocessed_data/pretrian_lookup_table'
pretrained_embeddings = '../pre_trained/glove.840B.300d.txt'
pretrained_embeddings_array = '../preprocessed_data/embeddings'
tokenized_captions = '../preprocessed_data/captions'


coco = COCO(caption_file)
captions = coco.anns

all_captions = []
for key in captions.keys():
    caption = nltk.word_tokenize(captions[key]['caption'])
    caption = [each.lower() for each in caption]
    all_captions.append(caption)

all_tokens = list(itertools.chain.from_iterable(all_captions))

unique_tokens, token_counts = np.unique(all_tokens, return_counts=True)

freq_idx = np.argsort(token_counts)[::-1]

token_counts = token_counts[freq_idx]
unique_tokens = unique_tokens[freq_idx]

word2count = {token: count for token, count in zip(unique_tokens, token_counts)}
word2idx = {token: idx for idx, token in enumerate(unique_tokens)}
idx2word = np.copy(unique_tokens)

pickle.dump({'word2count': word2count,
             'word2idx': word2idx,
             'idx2word': idx2word}, open(general_caption_info, 'wb'))



    
    

with open(pretrained_embeddings, 'r', encoding='utf-8') as f:
    embeddings = f.readlines()

all_embeddings = [np.zeros]
all_tokens = ['UNK']
for emb in embeddings:
    emb = emb.strip()
    emb = emb.split(' ')
    token = emb[0]
    embedding = np.asarray(emb[1:], dtype=np.float32)
    if embedding.shape[0] != 300 or token in all_tokens:
        continue
    all_embeddings.append(embedding)
    all_tokens.append(token)
    if len(all_tokens)>=500000:
        break


word2idx = {token: idx for idx, token in enumerate(all_tokens[:250000])}
idx2word = all_tokens[:250000].copy()
embeddings = np.stack(all_embeddings[:250000])


for key in captions.keys():
    caption = nltk.word_tokenize(captions[key]['caption'])
    caption = [each.lower() for each in caption]
    captions[key]['caption'] = [word2idx.get(token, word2idx['UNK']) for token in caption]

pickle.dump({'word2idx': word2idx,
             'idx2word': idx2word}, open(pretrian_lookup_table, 'wb'))

pickle.dump(embeddings, open(pretrained_embeddings_array, 'wb'))
pickle.dump(captions, open(tokenized_captions, 'wb'))












