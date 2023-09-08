import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, alpha=0.2):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(alpha),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(alpha),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(alpha),
            nn.Conv2d(features_d * 8, 1, 4, 2, 0),
        )

    def forward(self, x):
        return self.disc(x)
