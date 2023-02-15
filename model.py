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

    def _initialize_parameter(self):
        for layer in self.modules():
            if not isinstance(layer, nn.Linear): continue
            nn.init.normal_(layer.weight, 0, 0.02)
            break

    def forward(self, x):
        return torch.sigmoid(self.layers(x).squeeze())


class DCGenerator(nn.Module):
    def __init__(self, latent_dim, out_channel, channels):
        super().__init__()
        layers = []

        channels.append(out_channel)
        layers.append(nn.ConvTranspose2d(latent_dim, channels[0], 4, 1, 0, bias=False))
        for idx in range(1, len(channels)):
            layers.append(nn.ConvTranspose2d(channels[idx - 1], channels[idx], 4, 2, 1, bias=False))
            if idx != len(channels) - 1:
                layers.append(nn.BatchNorm2d(channels[idx], momentum=0.8))
                layers.append(nn.ReLU())
        layers.append(nn.Tanh())
        self.layer = nn.Sequential(*layers)
        self._initialize()
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.layer(x)


class DCDiscriminator(nn.Module):
    def __init__(self, in_channel, channels, dropout_rate):
        super().__init__()
        layers = []

        channels = [in_channel] + channels
        for idx in range(1, len(channels)):
            if idx != 1: layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.Conv2d(channels[idx - 1], channels[idx], 4, 2, 1, bias=False))
            if idx != 1: layers.append(nn.BatchNorm2d(channels[idx]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(channels[-1], 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())
        self.layer = nn.Sequential(*layers)
        self._initialize()
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.layer(x).squeeze(-1)
