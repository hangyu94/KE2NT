import torch
import torch.nn as nn


class TransformLoss(torch.nn.Module):

    def __init__(self):
        super(TransformLoss, self).__init__()
        self.loss_func = torch.nn.CosineSimilarity()

    def forward(self, x, y):      
        return 1. - self.loss_func(x, y)
        

class SelfConLoss(nn.Module):
   
    def __init__(self):
        super(SelfConLoss, self).__init__()

    def forward(self, anchor, pos, neg):
        anchor = anchor / anchor.norm(dim=-1, keepdim=True)    
        pos = pos / pos.norm(dim=-1, keepdim=True)   
        neg = neg / neg.norm(dim=-1, keepdim=True) 
  
        dp = anchor.float() @ pos.t()
        dn = anchor.float() @ neg.t()

        loss = dn - dp + 2

        return loss.mean()
