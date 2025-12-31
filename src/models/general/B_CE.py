import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.general.DenoisingRec import DenoisingRec

def loss_function(y, t):
    loss = F.binary_cross_entropy_with_logits(y, t, reduction='mean')
    return loss

class B_CE(DenoisingRec):
    def loss(self, out_dict):
        prediction = out_dict['prediction'].reshape(-1)
        label = out_dict['label'].float().reshape(-1)
        
        return loss_function(prediction, label)