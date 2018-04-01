import torch.nn as nn
import torch.nn.functional as F


# @ parameter
# channel: the channel of input image

class LeNet(nn.Module):
    def __init__(self, name, category, channel=1, size=32):
        super(LeNet, self).__init__()
        padding = 0
        self.name = name
        if size < 32:
            padding = (32 - size) // 2
        self.conv1 = nn.Conv2d(channel, 6, 5, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, category)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class modify_VGG(nn.Module):
    def __init__(self,name):
        super(modify_VGG, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
        )
        self.fc1=nn.Linear(256*4*4,4096)
        self.fc2=nn.Linear(4096,10)
        self.name = name

    def forward(self,x):
        x = self.layers(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


