import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models


# @ parameter
# channel: the channel of input image

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class LeNet(nn.Module):
    def __init__(self, name):
        super(LeNet, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # 28
        out = F.max_pool2d(out, 2)
        # 14
        out = F.relu(self.conv2(out))
        # 10
        out = F.max_pool2d(out, 2)
        # 5
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class CIFAR_Net(nn.Module):
    def __init__(self, name):
        super(CIFAR_Net, self).__init__()
        self.name = name
        self.layers = nn.Sequential(
            # 32
            nn.Conv2d(3, 96, 5),
            # 28
            nn.ReLU(True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(),
            # 14
            nn.Conv2d(96, 128, 5),
            # 10
            nn.ReLU(True),
            # 5
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout()

        )
        self.fc1 = nn.Linear(128 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.04)
                m.bias.data.fill_(0.1)


class modify_VGG(nn.Module):
    def __init__(self, name):
        super(modify_VGG, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 10)
        self.name = name
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.04)
                m.bias.data.fill_(0.1)


class m_LeNet(nn.Module):
    def __init__(self, name):
        super(m_LeNet, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # 28
        out = F.max_pool2d(out, 2)
        # 14
        out = F.relu(self.conv2(out))
        # 10
        out = F.max_pool2d(out, 2)
        # 5
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class m_CIFAR_Net(nn.Module):
    def __init__(self, name):
        super(m_CIFAR_Net, self).__init__()
        self.name = name
        self.layers = nn.Sequential(
            # 32
            nn.Conv2d(3, 96, 7),
            # 26
            nn.ReLU(True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(),
            # 13
            nn.Conv2d(96, 128, 7),
            # 7
            nn.ReLU(True),
            # 3
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout()

        )
        self.fc1 = nn.Linear(128 * 2 * 2, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.04)
                m.bias.data.fill_(0.1)
