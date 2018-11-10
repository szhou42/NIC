# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import torch
import numpy as np


def save_model(model_dir, epoch, batch_step_count, time_used_global, optimizer, encoder, decoder):
   state = {
            'epoch': epoch,
            'batch_step_count': batch_step_count,
            'time_used_global': time_used_global,
            'optimizer': optimizer.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
            }
   torch.save(state, open(model_dir + 'model_'+str(epoch)+'.pth', 'wb'))


def load_model(model_dir, model_list):
    lastest_model_idx = np.argmax([int(each_model.split('_')[1][:-4]) for each_model in model_list])
    lastest_model = model_dir + model_list[lastest_model_idx]
    lastest_state = torch.load(open(lastest_model, 'rb'))
    return lastest_state


# TO DO: sample images from valset and save it on tensorboard.
def save_images_and_captions(images, generated_captions, writer):
    im = images.cpu().numpy().transpose(0, 2, 3, 1)
    mean = [0.4701, 0.4469, 0.4076]
    std = [0.2692, 0.2646, 0.2801]

    im = np.array((im * std + mean) * 255, dtype=np.uint8)
    
    pass


def metrics():
    pass
