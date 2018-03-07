from torch.autograd import Variable
import torch
from loaddata import random_data
import torch.backends.cudnn as cudnn
import os

from net import LeNet
from loaddata import load


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
    print('Epoch {}/{}'.format(epoch, n_epochs))
    print("-" * 10)
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


# @parameter
# net: the architecture of training network
# testloader: the set of data which comes from load.py in package dataload
# cost: cost function

def test(net, testloader, cost, use_cuda):
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    print("test process")
    print("-" * 10)

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


# @parameter
# net: the architecture of training network
# test_set: the set of data which comes from random_data.py in package dataload
# cost: cost function

def test_random(net, test_set):
    net.eval()
    print("test process")
    print("-" * 10)

    total = test_set.size(0)
    result = {}

    x_test = Variable(test_set)
    output = net(x_test)
    _, pred = torch.max(output.data, 1)
    for i in pred:
        if i in result.keys():
            result[i] += 1
        else:
            result[i] = 0

    with open("result.txt","w") as f:
        for i in result.keys():
            f.write("label is {}, and {}% for all data \n".format(i, 100 * result[i] / total))


def main():
    use_cuda = torch.cuda.is_available()

    net = LeNet.LeNet(1, 28)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(1))
        cudnn.benchmark = True

    cost = torch.nn.CrossEntropyLoss()

    if os.path.exists("./netWeight/"+net.name()):
        net.load_state_dict(torch.load("./netWeight/"+net.name()))
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        n_epochs = 1
        train_set, test_set = load.get_MNIST()
        for i in range(n_epochs):
            train(net, train_set, cost, optimizer, i, n_epochs, use_cuda)
        os.mkdir("./netWeight/")
        torch.save(net.state_dict(), "./netWeight/" + net.name())

    test_set = random_data.get_random_data(5000000, 1, 128, 28)
    if use_cuda:
        test_set = test_set.cuda()
    test_random(net, test_set)


if __name__ == '__main__':
    main()
