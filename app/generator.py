import math
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, img_size):
        super(Generator, self).__init__()
        
        layers = []
        
        # Determinar quantas camadas são necessárias com base no tamanho da imagem
        num_blocks = int(math.log2(img_size)) - 3  # 3 é subtraído porque começamos com 8x8 (2^3) e queremos ir até img_size
        out_channels = features_g * (2 ** num_blocks)
        
        # Bloco inicial
        layers.append(self._block(z_dim, out_channels, 4, 1, 0))
        
        # Blocos intermediários
        for i in range(num_blocks):
            in_channels = out_channels
            out_channels = in_channels // 2
            layers.append(self._block(in_channels, out_channels, 4, 2, 1))
        
        # Camada final
        layers.append(nn.ConvTranspose2d(out_channels, channels_img, 4, 2, 1))
        layers.append(nn.Tanh())
        
        self.gen = nn.Sequential(*layers)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.gen(x)
