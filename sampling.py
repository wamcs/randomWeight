from torch.autograd import Variable
import torch
from loaddata import random_data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import os
import numpy as np

from net.net import *
import time
from loaddata import load

plt.switch_backend('agg')

pic_path = './s_pic/'


def train(net, dataloader, cost, optimizer, epoch, n_epochs, use_cuda):
    # the model of training
    net.train()
    running_loss = 0.0
    print("-" * 10)
    print('Epoch {}/{}'.format(epoch, n_epochs))
    num = 0
    for item in dataloader:
        num += len(item)
        for data in item:
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

    print("Loss {}".format(running_loss / len(num)))
    print("-" * 10)


def test_random_N(net, data_size, batch_size, channel, dim, use_cuda, mean, std):
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
            x_test = random_data.get_N_random_data(channel=channel, size=data_size - (time - 1) * batch_size, dim=dim,
                                                   mean=mean, std=std)
        else:
            x_test = random_data.get_N_random_data(channel=channel, size=batch_size, dim=dim, mean=mean, std=std)

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
    print(result)
    return result


def test_random_U(net, data_size, batch_size, channel, dim, use_cuda):
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
            x_test = random_data.get_U_random_data(channel=channel, size=data_size - (time - 1) * batch_size, dim=dim)
        else:
            x_test = random_data.get_U_random_data(channel=channel, size=batch_size, dim=dim)

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
def test1(times=6, std=1):
    print('-' * 10)
    print('test 1')
    result = []
    data_size = 1000000
    train_set, val_set, test_set = load.get_MNIST(False, None)
    for k in range(times):
        print('train model')
        use_cuda = torch.cuda.is_available()
        net = LeNet(str(k))
        cost = torch.nn.CrossEntropyLoss()
        net.train()
        n_epochs = 100
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        start = time.time()
        if use_cuda:
            net.cuda()
            cost.cuda()
            # net = torch.nn.DataParallel(net, device_ids=[2])
            # cost = torch.nn.DataParallel(cost, device_ids=[2])
            cudnn.benchmark = True
        for i in range(n_epochs):
            train(net=net,
                  dataloader=[train_set, val_set],
                  cost=cost,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda)
        end = time.time()
        print('train end, take {}s'.format(end - start))

        temp = test_random_N(net=net, data_size=data_size, batch_size=10000, channel=1, dim=28, use_cuda=use_cuda,
                             mean=0,
                             std=std)
        result.append(temp)

    with open("random_test_1_{}.txt".format(int(std)), "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(temp, 100 * item[temp] / data_size))

    plt.figure()
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('N(0,{}) {} times comparision'.format(std, times))
    x = np.linspace(0, 9, 10).astype(np.int)
    plt.xticks(x)
    plt.yticks(np.linspace(0, 100, 20))
    for j, item in enumerate(result):
        y = np.zeros(10)
        for i in item.keys():
            y[i] = 100 * item[i] / data_size
        plt.plot(x, y)

    # just suit the situation std = [1,0.001]
    plt.savefig(pic_path + 'test1_{}.png'.format(int(std)))
    plt.close()


def test2(times=6):
    print('-' * 10)
    print('test 2')
    result = []
    data_size = 1000000
    for k in range(times):
        print('train model')
        use_cuda = torch.cuda.is_available()
        net = LeNet(str(k))
        cost = torch.nn.CrossEntropyLoss()
        net.train()
        n_epochs = 100
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        start = time.time()
        if use_cuda:
            net.cuda()
            cost.cuda()
            # net = torch.nn.DataParallel(net, device_ids=[2])
            # cost = torch.nn.DataParallel(cost, device_ids=[2])
            cudnn.benchmark = True
        train_set, val_set, test_set = load.get_MNIST(False, None)
        for i in range(n_epochs):
            train(net=net,
                  dataloader=[train_set, val_set],
                  cost=cost,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda)
        end = time.time()
        print('train end, take {}s'.format(end - start))

        temp = test_random_U(net=net, data_size=data_size, batch_size=10000, channel=1, dim=28, use_cuda=use_cuda)
        result.append(temp)

    with open("random_test_2.txt", "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(temp, 100 * item[temp] / data_size))

    plt.figure()
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('U {} times comparision'.format(times))
    x = np.linspace(0, 9, 10).astype(np.int)
    plt.xticks(x)
    plt.yticks(np.linspace(0, 100, 20))
    for j, item in enumerate(result):
        y = np.zeros(10)
        for i in item.keys():
            y[i] = 100 * item[i] / data_size
        plt.plot(x, y)

    # just suit the situation std = [1,0.001]
    plt.savefig(pic_path + 'test2.png')
    plt.close()


# test different mean and std
def test3(times=1):
    print('-' * 10)
    print('test 3')
    print('train model')
    use_cuda = torch.cuda.is_available()
    net = LeNet('LeNet')
    cost = torch.nn.CrossEntropyLoss()
    net.train()
    n_epochs = 100
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start = time.time()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[2])
        # cost = torch.nn.DataParallel(cost, device_ids=[2])
        cudnn.benchmark = True
    train_set, val_set, test_set = load.get_MNIST(False, None)
    for i in range(n_epochs):
        train(net=net,
              dataloader=[train_set, val_set],
              cost=cost,
              optimizer=optimizer,
              epoch=i,
              n_epochs=n_epochs,
              use_cuda=use_cuda)
    end = time.time()
    print('train end, take {}s'.format(end - start))

    std = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    data_size = 1000000

    for k in range(times):
        result = []
        for i in std:
            result.append(
                test_random_N(net=net, data_size=data_size, batch_size=10000, channel=1, dim=28, use_cuda=use_cuda,
                              mean=0,
                              std=i))

        with open("random_test_3_{}.txt".format(k), "w") as f:
            for i, item in enumerate(result):
                f.write("-" * 10)
                f.write("\n number:{} \n".format(i))
                f.write("mean is 0, std is {} \n".format(std[i]))
                for temp in item.keys():
                    f.write("label is {}, and {}% for all data \n".format(temp, 100 * item[temp] / data_size))

        plt.figure()
        plt.xlabel('label')
        plt.ylabel('ratio')
        plt.title('different N comparision')
        x = np.linspace(0, 9, 10).astype(np.int)
        plt.xticks(x)
        plt.yticks(np.linspace(0, 100, 20))
        for j, item in enumerate(result):
            y = np.zeros(10)
            for i in item.keys():
                y[i] = 100 * item[i] / data_size
            plt.plot(x, y, label="{}".format(std[j]))
        plt.legend("upper right")
        plt.savefig(pic_path + 'test3_{}.png'.format(k))
        plt.close()


def plot_random_MNIST(mean=0, std=1):
    print('-' * 10)
    print('test 4')
    print('train model')
    use_cuda = torch.cuda.is_available()
    net = LeNet('LeNet')
    cost = torch.nn.CrossEntropyLoss()
    net.train()
    n_epochs = 100
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start = time.time()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[2])
        # cost = torch.nn.DataParallel(cost, device_ids=[2])
        cudnn.benchmark = True
    train_set, val_set, test_set = load.get_MNIST(False, None)
    for i in range(n_epochs):
        train(net=net,
              dataloader=[train_set, val_set],
              cost=cost,
              optimizer=optimizer,
              epoch=i,
              n_epochs=n_epochs,
              use_cuda=use_cuda)
    end = time.time()
    print('train end, take {}s'.format(end - start))

    data_size = 1000000
    temp1 = test_random_N(net=net, data_size=data_size, batch_size=10000,
                          channel=1, dim=28, use_cuda=use_cuda,
                          mean=mean, std=std)
    temp2 = test_random_U(net=net, data_size=data_size, batch_size=10000,
                          channel=1, dim=28, use_cuda=use_cuda)

    plt.figure(1)
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('U[-1,1] random input')
    x = np.linspace(0, 9, 10).astype(np.int)
    plt.xticks(x)
    plt.yticks(np.linspace(0, 100, 20))
    y = np.zeros(10)
    for i in temp1.keys():
        y[i] = 100 * temp1[i] / data_size
    plt.bar(x, y, width=0.35, edgecolor='white')

    for a, b in zip(x, y):
        plt.text(a + 0.3, b + 0.05, '%.2f' % b, ha='center', va='bottom')
    plt.savefig(pic_path + 'MNIST_U.png')
    plt.close()
    plt.figure(2)
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('N({},{}) random input'.format(mean, std))
    x = np.linspace(0, 9, 10).astype(np.int)
    plt.xticks(x)
    plt.yticks(np.linspace(0, 100, 20))
    y = np.zeros(10)
    for i in temp2.keys():
        y[i] = 100 * temp2[i] / data_size
    plt.bar(x, y, width=0.35, edgecolor='white')

    for a, b in zip(x, y):
        plt.text(a + 0.3, b + 0.05, '%.2f' % b, ha='center', va='bottom')
    plt.savefig(pic_path + 'MNIST_N.png')
    plt.close()


CIFAR = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def test4(times=6, std=1):
    print('-' * 10)
    print('test 4')
    result = []
    data_size = 1000000
    train_set, val_set, test_set = load.get_CIFAR(False, None)
    for k in range(times):
        print('train model')
        use_cuda = torch.cuda.is_available()
        net = CIFAR_Net(str(k))
        cost = torch.nn.CrossEntropyLoss()
        net.train()
        n_epochs = 100
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.001)
        start = time.time()
        if use_cuda:
            net.cuda()
            cost.cuda()
            # net = torch.nn.DataParallel(net, device_ids=[2])
            # cost = torch.nn.DataParallel(cost, device_ids=[2])
            cudnn.benchmark = True
        for i in range(n_epochs):
            train(net=net,
                  dataloader=[train_set, val_set],
                  cost=cost,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda)
        end = time.time()
        print('train end, take {}s'.format(end - start))

        temp = test_random_N(net=net, data_size=data_size, batch_size=1000, channel=3, dim=32, use_cuda=use_cuda,
                             mean=0,
                             std=std)
        result.append(temp)

    with open("random_test_4_{}.txt".format(int(std)), "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(CIFAR[temp], 100 * item[temp] / data_size))

    plt.figure()
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('N(0,{}) {} times comparision'.format(std, times))
    x = np.linspace(0, 9, 10).astype(np.int)
    plt.xticks(x, np.array(CIFAR), rotation=45)
    plt.yticks(np.linspace(0, 100, 20))
    for j, item in enumerate(result):
        y = np.zeros(10)
        for i in item.keys():
            y[i] = 100 * item[i] / data_size
        plt.plot(x, y)

    # just suit the situation std = [1,0.001]
    plt.savefig(pic_path + 'test4_{}.png'.format(int(std)))
    plt.close()


def test5(times=6):
    print('-' * 10)
    print('test 2')
    result = []
    data_size = 1000000
    train_set, val_set, test_set = load.get_CIFAR(False, None)
    for k in range(times):
        print('train model')
        use_cuda = torch.cuda.is_available()
        net = CIFAR_Net(str(k))
        cost = torch.nn.CrossEntropyLoss()
        net.train()
        n_epochs = 100
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.001)
        start = time.time()
        if use_cuda:
            net.cuda()
            cost.cuda()
            # net = torch.nn.DataParallel(net, device_ids=[2])
            # cost = torch.nn.DataParallel(cost, device_ids=[2])
            cudnn.benchmark = True
        for i in range(n_epochs):
            train(net=net,
                  dataloader=[train_set, val_set],
                  cost=cost,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda)
        end = time.time()
        print('train end, take {}s'.format(end - start))
        temp = test_random_U(net=net, data_size=data_size, batch_size=1000, channel=3, dim=32, use_cuda=use_cuda)
        result.append(temp)

    with open("random_test_5.txt", "w") as f:
        for i, item in enumerate(result):
            f.write("-" * 10)
            f.write("\n number:{} \n".format(i))
            for temp in item.keys():
                f.write("label is {}, and {}% for all data \n".format(CIFAR[temp], 100 * item[temp] / data_size))

    plt.figure()
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('U {} times comparision'.format(times))
    x = np.linspace(0, 9, 10).astype(np.int)
    plt.xticks(x, np.array(CIFAR), rotation=45)
    plt.yticks(np.linspace(0, 100, 20))
    for j, item in enumerate(result):
        y = np.zeros(10)
        for i in item.keys():
            y[i] = 100 * item[i] / data_size
        plt.plot(x, y)
    plt.savefig(pic_path + 'test5.png')
    plt.close()


# test different mean and std
def test6(times=1):
    print('-' * 10)
    print('test 6')
    print('train model')
    use_cuda = torch.cuda.is_available()
    net = CIFAR_Net('CIFAR_Net')
    cost = torch.nn.CrossEntropyLoss()
    net.train()
    n_epochs = 100
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.001)
    start = time.time()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[2])
        # cost = torch.nn.DataParallel(cost, device_ids=[2])
        cudnn.benchmark = True
    train_set, val_set, test_set = load.get_CIFAR(False, None)
    for i in range(n_epochs):
        train(net=net,
              dataloader=[train_set, val_set],
              cost=cost,
              optimizer=optimizer,
              epoch=i,
              n_epochs=n_epochs,
              use_cuda=use_cuda)
    end = time.time()
    print('train end, take {}s'.format(end - start))

    std = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    data_size = 1000000
    for k in range(times):
        result = []
        for i in std:
            result.append(
                test_random_N(net=net, data_size=data_size, batch_size=1000, channel=3, dim=32, use_cuda=use_cuda,
                              mean=0,
                              std=i))

        with open("random_test_6_{}.txt".format(k), "w") as f:
            for i, item in enumerate(result):
                f.write("-" * 10)
                f.write("\n number:{} \n".format(i))
                f.write("mean is 0, std is {} \n".format(std[i]))
                for temp in item.keys():
                    f.write("label is {}, and {}% for all data \n".format(CIFAR[temp], 100 * item[temp] / data_size))

        plt.figure()
        plt.xlabel('label')
        plt.ylabel('ratio')
        plt.title('different N comparision')
        x = np.linspace(0, 9, 10).astype(np.int)
        plt.xticks(x, np.array(CIFAR), rotation=45)
        plt.yticks(np.linspace(0, 100, 20))
        for j, item in enumerate(result):
            y = np.zeros(10)
            for i in item.keys():
                y[i] = 100 * item[i] / data_size
            plt.plot(x, y, label="{}".format(std[j]))
        plt.legend("upper right")
        plt.savefig(pic_path + 'test6_{}.png'.format(k))
        plt.close()


def plot_random_CIFAR(mean=0, std=1):
    print('-' * 10)
    print('test 4')
    print('train model')
    use_cuda = torch.cuda.is_available()
    net = CIFAR_Net('CIFAR_Net')
    cost = torch.nn.CrossEntropyLoss()
    net.train()
    n_epochs = 100
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.001)
    start = time.time()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[2])
        # cost = torch.nn.DataParallel(cost, device_ids=[2])
        cudnn.benchmark = True
    train_set, val_set, test_set = load.get_CIFAR(False, None)
    for i in range(n_epochs):
        train(net=net,
              dataloader=[train_set, val_set],
              cost=cost,
              optimizer=optimizer,
              epoch=i,
              n_epochs=n_epochs,
              use_cuda=use_cuda)
    end = time.time()
    print('train end, take {}s'.format(end - start))

    data_size = 1000000
    temp1 = test_random_N(net=net, data_size=data_size, batch_size=1000,
                          channel=3, dim=32, use_cuda=use_cuda,
                          mean=mean, std=std)
    temp2 = test_random_U(net=net, data_size=data_size, batch_size=1000,
                          channel=3, dim=32, use_cuda=use_cuda)

    plt.figure(1)
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('U[-1,1] random input')
    x = np.linspace(0, 9, 10).astype(np.int)
    y = np.zeros(10)
    plt.xticks(x, np.array(CIFAR), rotation=45)
    plt.yticks(np.linspace(0, 100, 20))
    for i in temp1.keys():
        y[i] = 100 * temp1[i] / data_size
    plt.bar(x, y, width=0.35, edgecolor='white', align='center')

    for a, b in zip(x, y):
        plt.text(a + 0.3, b + 0.05, '%.2f' % b, ha='center', va='bottom')
    plt.savefig(pic_path + 'CIFAR_U.png')
    plt.close()
    plt.figure(2)
    plt.xlabel('label')
    plt.ylabel('ratio')
    plt.title('N({},{}) random input'.format(mean, std))
    plt.xticks(x, np.array(CIFAR), rotation=45)
    plt.yticks(np.linspace(0, 100, 20))
    y = np.zeros(10)
    for i in temp2.keys():
        y[i] = 100 * temp2[i] / data_size
    plt.bar(x, y, width=0.35, edgecolor='white', align='center')

    for a, b in zip(x, y):
        plt.text(a + 0.3, b + 0.05, '%.2f' % b, ha='center', va='bottom')
    plt.savefig(pic_path + 'CIFAR_N.png')
    plt.close()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # test1 need train std = 1 and std = 0.001
    c = 64 / 128 / 128
    # test1(std=1)
    # test1(std=0.001)
    # test2()
    test3()
    # plot_random_MNIST(0, c)

    # test4(std=1)
    # test4(std=0.001)
    # test5()
    test6()
    # plot_random_CIFAR(0, c)


if __name__ == '__main__':
    main()
