import math
import torch.nn as nn

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
