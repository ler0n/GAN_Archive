import torch

from torch import nn


class P2PGeneratorLoss(nn.Module):
    def __init__(self, lmd):
        super().__init__()
        self.lmd = torch.tensor(lmd)
        self.loss = nn.BCELoss()

    def forward(self, disc_pred, generate_pred, truth):
        entorpy_loss = self.loss(disc_pred, torch.ones_like(disc_pred))
        l1_loss = torch.mean(torch.abs(generate_pred - truth))
        return entorpy_loss + self.lmd * l1_loss
    

class P2PDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
    
    def forward(self, real_pred, generate_pred):
        real_loss = self.loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.loss(generate_pred, torch.zeros_like(generate_pred))
        return real_loss + fake_loss