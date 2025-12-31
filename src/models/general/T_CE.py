import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.general.DenoisingRec import DenoisingRec

def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduction='none')

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update

class T_CE(DenoisingRec):
    def loss(self, out_dict):
        prediction = out_dict['prediction'].reshape(-1)
        label = out_dict['label'].float().reshape(-1)
        
        return loss_function(prediction, label, self.current_drop_rate)