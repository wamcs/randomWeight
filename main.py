from torch.autograd import Variable
import torch
from loaddata import random_data
import torch.backends.cudnn as cudnn
import os

from net import LeNet
from loaddata import load

net_root = './netWeight/'


# @parameter
# net: the architecture of training network
# dataloader: the set of data which comes from load.py in package dataload
# cost: cost function
# optimizer: the optimization
# epoch: the time of present training process
# n_epochs: the number of training process

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


# @parameter
# net: the architecture of training network
# testloader: the set of data which comes from load.py in package dataload
# cost: cost function

def test(net, testloader, cost, use_cuda):
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    print("-" * 10)
    print("test process")

    for data in testloader:
        x_test, y_test = data
        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()
        x_test, y_test = Variable(x_test), Variable(y_test)
        output = net(x_test)
        test_loss += cost(output, y_test).data[0]
        _, pred = torch.max(output.data, 1)  # pred: get the index of the max probability
        correct += pred.eq(y_test.data.view_as(pred)).sum()
        total += y_test.size(0)
    print("Loss {}, Acc {}".format(test_loss / len(testloader), 100 * correct / total))
    print("-" * 10)


# @parameter
# net: the architecture of training network
# test_set: the set of data which comes from random_data.py in package dataload
# cost: cost function

def test_random(net, data_size, batch_size, channel, dim, use_cuda):
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
            x_test = random_data.get_random_data(channel=channel, size=data_size - (time - 1) * batch_size, dim=dim)
        else:
            x_test = random_data.get_random_data(channel=channel, size=batch_size, dim=dim)

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

    with open("result.txt", "w") as f:
        for i in result.keys():
            f.write("label is {}, and {}% for all data \n".format(i, 100 * result[i] / data_size))


def test_constant(net, data_size, batch_size, channel, dim, use_cuda, constant):
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
            x_test = random_data.get_constant_value(constant=constant, channel=channel,
                                                    size=data_size - (time - 1) * batch_size, dim=dim)
        else:
            x_test = random_data.get_constant_value(constant=constant, channel=channel, size=batch_size, dim=dim)

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
    temp = {}
    for i in result.keys():
        temp[i] = 100 * result[i] / data_size
    with open("result_random.txt", "w") as f:
        for i in result.keys():
            f.write("label is {}, and {}% for all data \n".format(i, 100 * result[i] / data_size))


def test_random_image():
    use_cuda = torch.cuda.is_available()
    net = LeNet.LeNet(1, 28)

    if not os.path.exists(net_root):
        os.mkdir(path=net_root)
    path = net_root + net.name()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(1))
        cudnn.benchmark = True
        path += '_cuda'

    cost = torch.nn.CrossEntropyLoss()

    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        n_epochs = 250
        train_set, test_set = load.get_MNIST()
        for i in range(n_epochs):
            train(net=net,
                  dataloader=train_set,
                  cost=cost,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda)
        torch.save(net.state_dict(), path)
        print('successfully save weights')

    test_random(net=net, data_size=1000000, batch_size=10000, channel=1, dim=28, use_cuda=use_cuda)
    temp = []
    for i in range(-1,1,step=0.1):
        temp.append(test_constant(constant=i,net=net, data_size=1000000, batch_size=10000, channel=1, dim=28, use_cuda=use_cuda))
    with open("result_constant.txt", "w") as f:
        for item in temp:
            for i in item.keys():
                f.write("label is {}, and {}% for all data \n".format(i, item[i]))



if __name__ == '__main__':
    test_random_image()
