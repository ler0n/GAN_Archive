import torch
from torch import nn


class VanillaBlock(nn.Module):
    def __init__(self, layer_dims, dropout_rate):
        super(VanillaBlock, self).__init__()
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-2], layer_dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2))
            if dropout_rate != 0: layers.append(nn.Dropout1d(dropout_rate))
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BaseGenerator(nn.Module):
    def __init__(self, layer_dims):
        super(BaseGenerator, self).__init__()
        self.layers = VanillaBlock(layer_dims, 0)
    
    def forward(self, x):
        return torch.tanh(self.layers(x))


class BaseDiscriminator(nn.Module):
    def __init__(self, layer_dims, dropout_rate):
        super(BaseDiscriminator, self).__init__()
        self.layers = VanillaBlock(layer_dims, dropout_rate)
        self._initialize_parameter()

    def _initialize_parameter(self):
        for layer in self.modules():
            if not isinstance(layer, nn.Linear): continue
            nn.init.normal_(layer.weight, 0, 0.02)
            break

    def forward(self, x):
        return torch.sigmoid(self.layers(x).squeeze())
