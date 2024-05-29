import torch.utils.data as data
from scipy.io import loadmat
import os
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
from math import floor

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class DOMINOPlusPlusLoss_fast(_Loss):
        
        def _init_(self):
          super()._init_()
          self.cross_entropy = DiceCELoss(to_onehot_y=True, softmax=True, batch=batch_dice)
          
        def ce(self, input: torch.Tensor, target: torch.Tensor):
          ce_compute = DiceCELoss(to_onehot_y=True, softmax=True)
          return ce_compute(input, target) 
        
        def penalty(self, input: torch.Tensor, target: torch.Tensor, matrix_penalty: torch.Tensor): 
        
          n, c, h, w, z = input.size()
       
          target_new = torch.flatten(target) # b * 1 * 64 * 64 * 64 -> b*1*64*64*64
          target_new = F.one_hot(target_new.to(torch.int64), c).cuda() #NHWZ * C
          target_new = target_new.unsqueeze(1) #NHWZ * 1 * C
      
          outputs = torch.swapaxes(input, 0, 1) # C * N * HWZ
          outputs = torch.reshape(outputs,(c, n*h*w*z)).cuda() #  
          outputs = torch.swapaxes(outputs, 0, 1) #nhwz , c
          outputs = outputs.unsqueeze(2) #nhwz , c, 1
        
          m = nn.Softmax(dim=1)
          outputs_soft = m(outputs).float()
        
          matrix_penalty_rep = matrix_penalty.unsqueeze(0).repeat(n*h*w*z, 1, 1)
          penalty = torch.bmm(target_new.float(),matrix_penalty_rep).cuda()   # (1 x N) * (N x N) = 1 x N
          penalty_term = torch.bmm(penalty.float(),outputs_soft)
          
          beta = 1. * self.ce(input, target)
          beta = 10**(int(floor(np.log10(beta))))
        
          penalty_sum = beta*(torch.mean(penalty_term).cuda())
          
          return penalty_sum
          
        def stepsizes(self, global_step: int, maxiter: int):
        
          alpha0 = (1-global_step/maxiter)
          alpha1 = global_step/maxiter
          
          return alpha0, alpha1
          
        def forward(self, input: torch.Tensor, target: torch.Tensor, matrix_penalty: torch.Tensor):
          ce_total = self.ce(input, target)
          penalty_total = self.penalty(input, target, matrix_penalty) 
          
          alpha0, alpha1 = self.stepsizes(global_step, maxiter)
          
          total_loss: torch.Tensor = (alpha1*ce_total) + (alpha0*penalty_total) 

          return total_loss
    
class DOMINO(_Loss):
    
    def __init__(self,weight=None, size_average=True):
        super(DOMINO,self).__init__()
    
    def forward(self, outputs, targets, matrix_penalty=matrix_penalty, N_classes=N_classes, Npixels=Npixels, length_targets=length_targets, total_batch=total_batch):
        
        #currently I set them like this to do each data point rather than whole batch at once
        penalty_term = torch.zeros(total_batch,1)
        entropy_term = torch.zeros(total_batch,1)
        
        target_vector = torch.reshape(target, (length_targets, Npixels)) # B * P
        target_vector = F.one_hot(target_vector.to(torch.int64), N_classes) #int8 instead of int64, #B * P * N
        
        output_vector = torch.reshape(outputs, (length_targets, N_classes, Npixels)) #B * N * P
        output_vector = torch.swapaxes(outputs_vector, 0, 1) # N * B * P

        for i in range(length_targets): #B
            for j in range(len(target_vector)): #P
                target_vectorr = torch.flatten(target_vector[i:i+1, j:j+1, :]).float()  #1 * 1 * N = N
                target_vectorr = torch.reshape(target_vector, (1,N_classes)) # 1 x N
                
                output_vectorr = torch.flatten(output_vector[:, i:i+1, j:j+1]).float()  # N * 1 * 1
                output_vectorr = torch.reshape(output_vector, (N_classes,1)) # N x 1
                output_vectorr = torch.exp(output_vector)/torch.sum(torch.exp(output_vector)) #softmax
                
                penalty = torch.mm(target_vector,matrix_penalty)   # (1 x N) * (N x N) = 1 x N
                penalty_term = torch.cat((penalty_term,torch.mm(penalty.float(),output_vector)),-1)
        
        loss_diceCE = DiceCELoss(to_onehot_y=True, softmax=True)
        entropy_term = loss_diceCE(outputs,targets)
        beta = 1.
        total_loss = entropy_term + beta*(torch.sum(penalty_term)/total_batch)
        return total_loss