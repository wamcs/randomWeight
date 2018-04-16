from torch.autograd import Variable
import torch
from loaddata import random_data
import torch.backends.cudnn as cudnn
import os
import numpy as np

from net.net import *
import time
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


def test_random(net, data_size, batch_size, channel, dim, use_cuda, mean, std):
    net.eval()
    print("-" * 10)
    print("test process")
    print("-" * 10)

    result = {}
    if data_size % batch_size == 0:
        time = data_size // batch_size
    else:
        time = data_size // batch_size + 1

    for i in range(time):
        print('time {}/{}'.format(i + 1, time))
        if i == time - 1:
            x_test = random_data.get_random_data(channel=channel, size=data_size - (time - 1) * batch_size, dim=dim,
                                                 mean=mean, std=std)
        else:
            x_test = random_data.get_random_data(channel=channel, size=batch_size, dim=dim, mean=mean, std=std)

        if use_cuda:
            x_test = x_test.cuda()
        x_test = Variable(x_test)
        output = net(x_test)
        _, pred = torch.max(output.data, 1)
        pred = pred.view(-1)
        for j in pred:
            if j in result.keys():
                result[j] += 1
            else:
                result[j] = 0
    return result


# repeat experiment, to check the classified situation of random image(MNIST)
def test1(times=3):
    print('-' * 10)
    print('test 1')
    result = []
    data_size = 10000
    for i in range(times):
        print('train model')
        use_cuda = torch.cuda.is_available()
        net = LeNet('LeNet')
        cost = torch.nn.CrossEntropyLoss()
        net.train()
        n_epochs = 250
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        start = time.time()
        if use_cuda:
            net.cuda()
            cost.cuda()
            # net = torch.nn.DataParallel(net, device_ids=[2])
            # cost = torch.nn.DataParallel(cost, device_ids=[2])
            cudnn.benchmark = True
        train_set, test_set = load.get_MNIST(False, None)
        for i in range(n_epochs):
            train(net=net,
                  dataloader=train_set,
                  cost=cost,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda)
        end = time.time()
        print('train end, take {}s'.format(end - start))

        temp = test_random(net=net, data_size=data_size, batch_size=200, channel=1, dim=28, use_cuda=use_cuda, mean=0,
                           std=1)
        result.append(temp)

    with open("random_test_1.txt", "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(temp, 100 * item[temp] / data_size))


# test different mean and std
def test2():
    print('-' * 10)
    print('test 2')
    print('train model')
    use_cuda = torch.cuda.is_available()
    net = LeNet('LeNet')
    cost = torch.nn.CrossEntropyLoss()
    net.train()
    n_epochs = 250
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    start = time.time()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[2])
        # cost = torch.nn.DataParallel(cost, device_ids=[2])
        cudnn.benchmark = True
    train_set, test_set = load.get_MNIST(False, None)
    for i in range(n_epochs):
        train(net=net,
              dataloader=train_set,
              cost=cost,
              optimizer=optimizer,
              epoch=i,
              n_epochs=n_epochs,
              use_cuda=use_cuda)
    end = time.time()
    print('train end, take {}s'.format(end - start))

    std = [1, 0.1, 0.01, 0.001]
    data_size = 10000
    result = []
    for i in std:
        result.append(
            test_random(net=net, data_size=data_size, batch_size=200, channel=1, dim=28, use_cuda=use_cuda, mean=0,
                        std=i))

    with open("random_test_2.txt", "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            f.write("mean is 0, std is {} \n".format(std[i]))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(temp, 100 * item[temp] / data_size))


# repeat experiment, to check the classified situation of random image(CIFAR)
def test3(times=3):
    print('-' * 10)
    print('test 3')
    result = []
    data_size = 10000
    for i in range(times):
        print('train model')
        use_cuda = torch.cuda.is_available()
        net = modify_VGG('modify_VGG')
        cost = torch.nn.CrossEntropyLoss()
        net.train()
        n_epochs = 300
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        start = time.time()
        if use_cuda:
            net.cuda()
            cost.cuda()
            # net = torch.nn.DataParallel(net, device_ids=[2])
            # cost = torch.nn.DataParallel(cost, device_ids=[2])
            cudnn.benchmark = True
        train_set, test_set = load.get_CIFAR(False, None)
        for i in range(n_epochs):
            train(net=net,
                  dataloader=train_set,
                  cost=cost,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda)
        end = time.time()
        print('train end, take {}s'.format(end - start))

        temp = test_random(net=net, data_size=data_size, batch_size=200, channel=3, dim=32, use_cuda=use_cuda, mean=0,
                           std=1)
        result.append(temp)

    with open("random_test_3.txt", "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(temp, 100 * item[temp] / data_size))


# test different mean and std
def test4():
    print('-' * 10)
    print('test 4')
    print('train model')
    use_cuda = torch.cuda.is_available()
    net = modify_VGG('modify_VGG')
    cost = torch.nn.CrossEntropyLoss()
    net.train()
    n_epochs = 300
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    start = time.time()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[2])
        # cost = torch.nn.DataParallel(cost, device_ids=[2])
        cudnn.benchmark = True
    train_set, test_set = load.get_CIFAR(False, None)
    for i in range(n_epochs):
        train(net=net,
              dataloader=train_set,
              cost=cost,
              optimizer=optimizer,
              epoch=i,
              n_epochs=n_epochs,
              use_cuda=use_cuda)
    end = time.time()
    print('train end, take {}s'.format(end - start))

    std = [1, 0.1, 0.01, 0.001]
    data_size = 10000
    result = []
    for i in std:
        result.append(
            test_random(net=net, data_size=data_size, batch_size=200, channel=3, dim=32, use_cuda=use_cuda, mean=0,
                        std=i))

    with open("random_test_4.txt", "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            f.write("mean is 0, std is {} \n".format(std[i]))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(temp, 100 * item[temp] / data_size))


def main():
    test1()
    test2()
    test3()
    test4()

if __name__ == '__main__':
    main()