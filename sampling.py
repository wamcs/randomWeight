from torch.autograd import Variable
import torch
from loaddata import random_data
import torch.backends.cudnn as cudnn
import os
import numpy as np

from net.net import *
from loaddata import load


def train(net, dataloader, cost, optimizer, epoch, n_epochs, use_cuda):
    # the model of training
    net.train()
    running_loss = 0.0
    print("-" * 10)
    print('Epoch {}/{}'.format(epoch, n_epochs))
    for data in dataloader:
        x_train, y_train = data
        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        # change data to Variable, a wrapper of Tensor
        x_train, y_train = Variable(x_train), Variable(y_train)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(x_train)
        loss = cost(outputs, y_train)
        loss.backward()
        # optimize the weight of this net
        optimizer.step()

        running_loss += loss.data[0]

    print("Loss {}".format(running_loss / len(dataloader)))
    print("-" * 10)


def