from torch.autograd import Variable
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from net.net import *
from loaddata import load
from loaddata.p_creator import *

net_root = './netWeight/'
pic_root = './pic/'
'this variable used to control whether retrain net'
net_type = {'M0': False, 'M1': True, 'M2': True}
net_epochs = {'MNIST': 100, 'CIFAR': 100, 'ImageNet': 150}

line = {0: '-', 1: '--', 2: ':'}
plt.switch_backend('agg')

'''
  @parameter
  net: the architecture of training network
  dataloader: the set of data which comes from load.py in package dataload
  cost: cost function
  optimizer: the optimization
  epoch: the time of present training process
  n_epochs: the number of training process
'''


def train(net, dataloader, cost, optimizer, epoch, n_epochs, use_cuda):
    # the model of training
    net.train()
    running_loss = 0.0
    correct = 0.0
    total = 0
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
        if use_cuda:
            _, pred = torch.max(outputs.data.cpu(), 1)  # pred: get the index of the max probability
            correct += pred.eq(y_train.data.cpu().view_as(pred)).sum()
        else:
            _, pred = torch.max(outputs.data, 1)
            correct += pred.eq(y_train.data.view_as(pred)).sum()
        total += y_train.size(0)
    Loss = running_loss / len(dataloader)
    train_acc = 100 * correct / total
    print("Loss {}, acc {}".format(Loss, train_acc))
    print("-" * 10)
    return Loss, train_acc


'''
  @parameter
  net: the architecture of training network
  testloader: the set of data which comes from load.py in package dataload
  cost: cost function
'''


def test(net, testloader, cost, use_cuda):
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    # temp = []
    for data in testloader:
        x_test, y_test = data
        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()
        x_test, y_test = Variable(x_test), Variable(y_test)
        output = net(x_test)
        # if use_cuda:
        #     temp.append(output.data.cpu().numpy())
        # else:
        #     temp.append(output.data.numpy())
        test_loss += cost(output, y_test).data[0]
        if use_cuda:
            _, pred = torch.max(output.data.cpu(), 1)  # pred: get the index of the max probability
            correct += pred.eq(y_test.data.cpu().view_as(pred)).sum()
        else:
            _, pred = torch.max(output.data, 1)
            correct += pred.eq(y_test.data.view_as(pred)).sum()

        total += y_test.size(0)

    Loss = test_loss / len(testloader)
    acc = 100 * correct / total
    print("Loss {}, Acc {}".format(Loss, acc))
    print("-" * 10)
    return Loss, acc


def train_model(net, cost, optimizer, n_epochs, train_set, val_set, use_cuda, type, index, retrain=False):
    print('train model')
    print(use_cuda)
    if not os.path.exists(net_root):
        os.mkdir(net_root)
    path = net_root + net.name + '_' + type + '_' + str(index)
    net.train()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[0,1])
        # cost = torch.nn.DataParallel(cost, device_ids=[0,1])
        cudnn.benchmark = True
        path += '_cuda'

    if not retrain:
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            return 0, np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)

    start = time.time()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for i in range(n_epochs):
        t_loss, t_acc = train(net=net,
                              dataloader=train_set,
                              cost=cost,
                              optimizer=optimizer,
                              epoch=i,
                              n_epochs=n_epochs,
                              use_cuda=use_cuda)
        print('val:')
        v_loss, v_acc = test(net=net,
                             testloader=val_set,
                             cost=cost,
                             use_cuda=use_cuda)
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
    end = time.time()
    torch.save(net.state_dict(), path)
    print('successfully save weights, take {}s'.format(end - start))
    return end - start, np.array(train_loss), np.array(train_acc), np.array(val_loss), np.array(val_acc)


def test_model(net, cost, test_set, use_cuda):
    print('test model')
    net.eval()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[2])
        # cost = torch.nn.DataParallel(cost, device_ids=[2])
        cudnn.benchmark = True
    start = time.time()
    loss, acc = test(net=net,
                     testloader=test_set,
                     cost=cost,
                     use_cuda=use_cuda)
    end = time.time()
    print('test_finish, take {}s'.format(end - start))
    return end - start, loss, acc


'''
net type:
M0: original data
M1: permute row and column of all images in the same order
M2: concatenate the image matrix as a long vector, and permute the vectors in the same order.

'''


def MNIST(times=3, retrain=False):
    use_cuda = torch.cuda.is_available()
    reuse = False
    statistic = {}
    for key in net_type.keys():
        statistic[key] = []  # time,train time, test time, test loss, test accuracy
    if not retrain:
        if os.path.exists('MNIST.csv'):
            ps = np.loadtxt('MNIST.csv', delimiter=',').astype(int)
            reuse = True
        else:
            ps = []
    else:
        ps = []
    for i in range(times):
        plt.figure(1)
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        plt.title('training loss vs validation loss ' + str(i))
        plt.figure(2)
        plt.xlabel('epochs')
        plt.ylabel('accuracy probability')
        plt.title('training accuracy vs validation accuracy ' + str(i))
        for j, key in enumerate(net_type.keys()):
            cost = torch.nn.CrossEntropyLoss()  # since pi from softmax function, this Loss is softmax Loss
            temp = []
            model = LeNet(name='MNIST_net')
            epochs = net_epochs['MNIST']
            start = time.time()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            if key == 'M0':
                train_set, val_set, test_set = load.get_MNIST(random=False, p=None)
            if key == 'M1':
                if reuse:
                    p = ps[i * 2 + j - 1]
                else:
                    p = M1(28)
                    ps.append(p)
                train_set, val_set, test_set = load.get_MNIST(random=True, p=p)
            if key == 'M2':
                if reuse:
                    p = ps[i * 2 + j - 1]
                else:
                    p = M2(28)
                    ps.append(p)
                train_set, val_set, test_set = load.get_MNIST(random=True, p=p)

            train_time, train_loss, train_acc, val_loss, val_acc = train_model(net=model,
                                                                               cost=cost,
                                                                               optimizer=optimizer,
                                                                               n_epochs=epochs,
                                                                               train_set=train_set,
                                                                               val_set=val_set,
                                                                               use_cuda=use_cuda,
                                                                               type=key,
                                                                               index=i,
                                                                               retrain=retrain)

            x = np.linspace(0, epochs, epochs)
            plt.figure(1)
            plt.plot(x, train_loss, 'g' + line[j], label='M' + str(j) + ' training loss')
            plt.plot(x, val_loss, 'b' + line[j], label='M' + str(j) + ' validation loss')
            plt.legend(loc='upper right')
            plt.figure(2)
            plt.plot(x, train_acc, 'g' + line[j], label='M' + str(j) + ' training accuracy')
            plt.plot(x, val_acc, 'b' + line[j], label='M' + str(j) + ' validation accuracy')
            plt.legend(loc='lower right')

            end = time.time()
            temp.append(start - end)
            temp.append(train_time)
            test_time, loss, acc = test_model(net=model,
                                              cost=cost,
                                              test_set=test_set,
                                              use_cuda=use_cuda)
            temp.append(test_time)
            temp.append(loss)
            temp.append(acc)
            temp.append(train_loss[-1])
            temp.append(train_acc[-1])
            temp.append(val_loss[-1])
            temp.append(val_acc[-1])
            statistic[key].append(temp)
        plt.figure(1)
        plt.savefig(pic_root + 'MNIST_' + str(i) + '_loss.png')
        plt.close()
        plt.figure(2)
        plt.savefig(pic_root + 'MNIST_' + str(i) + '_acc.png')
        plt.close()
    if not reuse:
        ps = np.array(ps)
        np.savetxt('MNIST.csv', ps, delimiter=',')
    with open("MNIST.txt", "w") as f:
        for key in statistic.keys():
            f.write('type is {} \n'.format(key))
            for item in statistic[key]:
                f.write(
                    'this process spends totally {}s, train spends {}s, test spends {}s, '
                    'test loss is {}, test accuracy is {}; train loss is {}, train accuracy is {}'
                    'validation loss is {}, validation accuracy is {}\n'.format(
                        item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]))


def CIFAR(times=3, retrain=False):
    use_cuda = torch.cuda.is_available()
    reuse = False
    statistic = {}
    for key in net_type.keys():
        statistic[key] = []  # time,train time, test time, test loss, test accuracy
    if not retrain:
        if os.path.exists('CIFAR.csv'):
            ps = np.loadtxt('CIFAR.csv', delimiter=',').astype(int)
            reuse = True
        else:
            ps = []
    else:
        ps = []
    for i in range(times):
        plt.figure(1)
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        plt.title('training loss vs validation loss ' + str(i))
        plt.figure(2)
        plt.xlabel('epochs')
        plt.ylabel('accuracy probability')
        plt.title('training accuracy vs validation accuracy ' + str(i))
        for j, key in enumerate(net_type.keys()):
            cost = torch.nn.CrossEntropyLoss()  # since pi from softmax function, this Loss is softmax Loss
            temp = []
            model = CIFAR_Net(name='CIFAR_net')
            epochs = net_epochs['CIFAR']
            start = time.time()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
            if key == 'M0':

                train_set, val_set, test_set = load.get_CIFAR(random=False, p=None)
            if key == 'M1':
                if reuse:
                    p = ps[i * 2 + j - 1]
                else:
                    p = M1(32)
                    ps.append(p)
                train_set, val_set, test_set = load.get_CIFAR(random=True, p=p)
            if key == 'M2':

                if reuse:
                    p = ps[i * 2 + j - 1]
                else:
                    p = M2(32)
                    ps.append(p)
                train_set, val_set, test_set = load.get_CIFAR(random=True, p=p)
            train_time, train_loss, train_acc, val_loss, val_acc = train_model(net=model,
                                                                               cost=cost,
                                                                               optimizer=optimizer,
                                                                               n_epochs=epochs,
                                                                               train_set=train_set,
                                                                               val_set=val_set,
                                                                               use_cuda=use_cuda,
                                                                               type=key,
                                                                               index=i,
                                                                               retrain=retrain)

            x = np.linspace(0, epochs, epochs)
            plt.figure(1)
            plt.plot(x, train_loss, 'g' + line[j], label='M' + str(j) + ' training loss')
            plt.plot(x, val_loss, 'b' + line[j], label='M' + str(j) + ' validation loss')
            plt.legend(loc='upper right')
            plt.figure(2)
            plt.plot(x, train_acc, 'g' + line[j], label='M' + str(j) + ' training accuracy')
            plt.plot(x, val_acc, 'b' + line[j], label='M' + str(j) + ' validation accuracy')
            plt.legend(loc='lower right')

            end = time.time()
            temp.append(start - end)
            temp.append(train_time)
            test_time, loss, acc = test_model(net=model,
                                              cost=cost,
                                              test_set=test_set,
                                              use_cuda=use_cuda)
            temp.append(test_time)
            temp.append(loss)
            temp.append(acc)
            temp.append(train_loss[-1])
            temp.append(train_acc[-1])
            temp.append(val_loss[-1])
            temp.append(val_acc[-1])
            statistic[key].append(temp)

        plt.figure(1)
        plt.savefig(pic_root + 'CIFAR_' + str(i) + '_loss.png')
        plt.close()
        plt.figure(2)
        plt.savefig(pic_root + 'CIFAR_' + str(i) + '_acc.png')
        plt.close()
    if not reuse:
        ps = np.array(ps)
        np.savetxt('CIFAR.csv', ps, delimiter=',')
    with open("CIFAR.txt", "w") as f:
        for key in statistic.keys():
            f.write('type is {} \n'.format(key))
            for item in statistic[key]:
                f.write(
                    'this process spends totally {}s, train spends {}s, test spends {}s, '
                    'test loss is {}, test accuracy is {}; train loss is {}, train accuracy is {}'
                    'validation loss is {}, validation accuracy is {}\n'.format(
                        item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]))


def ImageNet(category, times=1, retrain=False):
    use_cuda = torch.cuda.is_available()
    reuse = False
    statistic = {}
    if not retrain:
        if os.path.exists('ImageNet_{}.csv'.format(category)):
            ps = np.loadtxt('ImageNet_{}.csv'.format(category), delimiter=',').astype(int)
            reuse = True
        else:
            ps = []
    else:
        ps = []
    for i, key in enumerate(net_type.keys()):
        temps = []  # time,train time, test time, test loss, test accuracy
        for j in range(times):
            cost = torch.nn.CrossEntropyLoss()  # since pi from softmax function, this Loss is softmax Loss
            temp = []
            model = models.vgg19(False, num_classes=category)
            model.name = 'VGG' + str(category)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            epochs = net_epochs['ImageNet']
            start = time.time()

            if key == 'M0':
                train_set, val_set, test_set = load.get_imageNet(random=False, p=None, category=category)
            if key == 'M1':
                if reuse:
                    p = ps[(i - 1) * times + j + 1]
                else:
                    p = M1(224)
                    ps.append(p)
                train_set, val_set, test_set = load.get_imageNet(random=True, p=p, category=category)
            if key == 'M2':
                if reuse:
                    p = ps[(i - 1) * times + j + 1]
                else:
                    p = M2(224)
                    ps.append(p)
                train_set, val_set, test_set = load.get_imageNet(random=True, p=p, category=category)

            train_time, train_loss, train_acc, val_loss, val_acc = train_model(net=model,
                                                                               cost=cost,
                                                                               optimizer=optimizer,
                                                                               n_epochs=epochs,
                                                                               train_set=train_set,
                                                                               val_set=val_set,
                                                                               use_cuda=use_cuda,
                                                                               type=key,
                                                                               index=j,
                                                                               retrain=retrain)
            end = time.time()
            temp.append(start - end)
            temp.append(train_time)
            test_time, loss, acc = test_model(net=model,
                                              cost=cost,
                                              test_set=test_set,
                                              use_cuda=use_cuda)
            temp.append(test_time)
            temp.append(loss)
            temp.append(acc)
            temps.append(temp)

        statistic[key] = temps
    if not reuse:
        ps = np.array(ps)
        np.savetxt('ImageNet_{}.csv'.format(category), ps, delimiter=',')
    with open('ImageNet_{}.txt'.format(category), "w") as f:
        for key in statistic.keys():
            f.write('type is {} \n'.format(key))
            for item in statistic[key]:
                f.write(
                    'this process spends totally {}s, train spends {}s, test spends {}s, '
                    'test loss is {} and test accuracy is {}\n'.format(
                        item[0], item[1], item[2], item[3], item[4]))


def main():
    MNIST(3)
    # CIFAR(3)
    # ImageNet(10)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
