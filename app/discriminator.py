import math
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, alpha, img_size):
        super(Discriminator, self).__init__()
        
        layers = []
        
        num_blocks = int(math.log2(img_size)) - 3
        
        layers.append(nn.Conv2d(channels_img, features_d, 4, 2, 1))
        layers.append(nn.LeakyReLU(alpha))
        
        in_channels = features_d
        for i in range(num_blocks - 1):
            out_channels = in_channels * 2
            layers.append(self._block(in_channels, out_channels, alpha))
            in_channels = out_channels

        layers.append(nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1))
        layers.append(nn.LeakyReLU(alpha))
        layers.append(nn.Conv2d(in_channels * 2, 1, 4, 2, 0))
        
        self.disc = nn.Sequential(*layers)

    def _block(self, in_channels, out_channels, alpha):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        return self.disc(x)
