import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.general.DenoisingRec import DenoisingRec

def loss_function(y, t, alpha):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
    y_ = torch.sigmoid(y).detach()
    weight = torch.pow(y_, alpha) * t + torch.pow((1-y_), 1-alpha) * (1-t)
    loss_ = loss * weight
    loss_ = torch.mean(loss_)
    return loss_

class R_CE(DenoisingRec):
    def loss(self, out_dict):
        prediction = out_dict['prediction'].reshape(-1)
        label = out_dict['label'].float().reshape(-1)
        
        return loss_function(prediction, label, self.alpha)