import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from functools import partial
from torchmetrics.functional import concordance_corrcoef

class BalanceBCELoss(nn.Module):
    def __init__(self,):
        

class Tilde_Q(nn.Module):
    def __init__(self,alpha=0.99,gamma=1):
        super(Tilde_Q, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma 
    def forward(self, input, target):
        '''
        input: (batch_size,sequence_length)
        target: (batch_size,sequence_length)
        '''
        bacth_size = input.size(0)
        sequence_length = input.size(1)
        shift_loss = sequence_length * F.smooth_l1_loss(
            torch.ones_like(input)*1.0/sequence_length,
            F.softmax(input-target,dim = 1)
        )

        # same main frquency 
        freq_input = torch.fft.rfft(input, dim = 1,norm="ortho").abs()
        freq_target = torch.fft.rfft(target, dim = 1,norm="ortho").abs()
        phase_loss = F.smooth_l1_loss(
                torch.ones_like(freq_input.argmax()),
                freq_input.argmax()/freq_target.argmax()
        )
        corr_loss = concordance_corrcoef(input,target)
        corr_loss = F.smooth_l1_loss(
                torch.ones_like(corr_loss),
                corr_loss
        )
        loss =  self.alpha * shift_loss + (1.0 - self.alpha)* phase_loss + self.gamma * corr_loss
        return loss
    def neg_loss(self, input, target):
        return -self(input,target)