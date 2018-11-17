# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage import io
#from nltk.translate.bleu_score import sentence_bleu

def save_model(model_name, model_dir, epoch, batch_step_count, time_used_global, optimizer, encoder, decoder):
   state = {
            'epoch': epoch,
            'batch_step_count': batch_step_count,
            'time_used_global': time_used_global,
            'optimizer': optimizer.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
            }
   torch.save(state, open(model_dir + model_name + '_' + str(epoch)+'.pth', 'wb'))


def load_model(model_dir, model_list):
    lastest_model_idx = np.argmax([int(each_model.split('_')[1][:-4]) for each_model in model_list])
    lastest_model = model_dir + model_list[lastest_model_idx]
    lastest_state = torch.load(open(lastest_model, 'rb'))
    return lastest_state


# TODO: sample images from valset and save it on tensorboard. Not finished yet.
def save_images_and_captions(image, generated_captions, writer):
    im = image.cpu().numpy().transpose(0, 2, 3, 1)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im = np.array((im * std + mean) * 255, dtype=np.uint8)
    io.imshow(im[0])
    
    pass

# TODO: Not finished yet.
def generate_caption(encoder, decoder, transform_val):
    encoder.eval()
    decoder.eval()
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open('../test_images/test8.jpg')
    image = transform_val(image)
    image = image.expand([1, -1, -1, -1])
    image = image.cuda()
    image_embeddings = encoder(image)
    with torch.no_grad():                       
#        caption = decoder.beam_search_generator(image_embeddings)
        caption = decoder.greedy_generator(image_embeddings)
        print(' '.join(caption[0]))
        caption, score = decoder.beam_search_generator_v2(image_embeddings, 0)
        print(' '.join(caption[0][0]))
        caption = decoder.beam_search_generator(image_embeddings)
        print(' '.join(caption[0]))
    return caption
    

'''
bleu score, specify which BLEU scores to use (e.g type can be 1,2,3, or 4)
predicted_sentences is a list of sentences
true_sentences is also a list of sentences
def bleu_score(type, predicted_sentences, true_sentences):
    weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33), (0.25,0.25,0.25,0.25)]
    weight = weights[type - 1]
    score_sum = 0
    for i in range(len(predicted_sentences)):
        score_sum = score_sum + sentence_bleu(true_sentences[i], predicted_sentences[i], weights = weight)
    score_avg = score_sum / len(predicted_sentences) 
    return score_avg
'''
