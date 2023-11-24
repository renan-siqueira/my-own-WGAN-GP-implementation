import math
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        _, C, H, W = x.size()
        query = self.query(x).view(-1, H * W).permute(0, 2, 1)
        key = self.key(x).view(-1, H * W)
        value = self.value(x).view(-1, H * W)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(-1, C, H, W)

        return out + x

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, img_size):
        super(Generator, self).__init__()

        layers = []

        num_blocks = int(math.log2(img_size)) - 3
        out_channels = features_g * (2 ** num_blocks)

        layers.append(nn.ConvTranspose2d(z_dim, out_channels, 4, 1, 0))
        layers.append(nn.ReLU(True))

        for i in range(num_blocks):
            in_channels = out_channels
            out_channels = in_channels // 2
            layers.append(self._block(in_channels, out_channels, 4, 2, 1))
            
            if i == num_blocks // 2:  # Adiciona a camada de atenção na metade do gerador
                layers.append(SelfAttention(out_channels))
            
            # Adiciona camada residual em cada bloco
            layers.append(ResidualBlock(out_channels))

        layers.append(nn.ConvTranspose2d(out_channels, channels_img, 4, 2, 1))
        layers.append(nn.Tanh())

        self.gen = nn.Sequential(*layers)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.gen(x)