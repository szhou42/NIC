# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import numpy as np

import torch


def save_model(epoch, time_used_global, optimizer, encoder, decoder):
   state = {
            'epoch': epoch,
            'time_used_global': time_used_global,
            'optimizer': optimizer.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
            }
   torch.save(state, open('model_'+str(epoch)+'.pth', 'wb'))

def load_model(model_dir, model_list):
    lastest_model_idx = np.argmax([int(each_model.split('_')[1][:-4]) for each_model in model_list])
    lastest_model = model_dir + model_list[lastest_model_idx]
    lastest_state = torch.load(open(lastest_model, 'rb'))
    return lastest_state



