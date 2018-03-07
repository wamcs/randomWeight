import torch.nn as nn
import torch.nn.functional as F

vggtypes = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M']}


class VGG(nn.Module):
    def __init__(self, vggtype):
        super(VGG, self).__init__()
        self.layer = self.generate_layer(vggtype)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def generate_layer(self, type):
        layers = []
        in_channels = 3
        for i in vggtypes[type]:
            if i == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, i, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(i))
                layers.append(nn.ReLU(inplace=True))
                in_channels = i
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg11():
    return VGG('VGG11')


def vgg13():
    return VGG('VGG13')


def vgg16():
    return VGG('VGG16')


def vgg19():
    return VGG('VGG19')