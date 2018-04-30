from torch.autograd import Variable
import torch
import torch.backends.cudnn as cudnn
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from net.net import *
from loaddata import load
from loaddata.p_creator import *

'this variable used to control whether retrain net'
net_type = ['M0', 'M1', 'M2']
net_epochs = {'MNIST': 100, 'CIFAR': 100}

line = {0: '-', 1: '--', 2: ':'}
plt.switch_backend('agg')
pic_path = './a_pic/'

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


def train_model(net, cost, optimizer, n_epochs, train_set, val_set, use_cuda):
    print('train model')
    print(use_cuda)

    net.train()
    if use_cuda:
        net.cuda()
        cost.cuda()
        # net = torch.nn.DataParallel(net, device_ids=[0,1])
        # cost = torch.nn.DataParallel(cost, device_ids=[0,1])
        cudnn.benchmark = True

    start = time.time()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    paras = []
    norm = []
    angle = []
    temps = []
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
        temp = []
        for i, para in enumerate(net.parameters()):
            temp.append(para.view(1, -1).data.cpu().numpy()[0])
        temp = np.concatenate(tuple(temp))
        temps.append(temp)
        print(temp.shape)
        if paras == []:
            paras = temp
            norm.append(0)
            angle.append(0)
        else:
            a = temp - paras
            print('norm', np.linalg.norm(a, ord=2) / len(a))
            norm.append(np.linalg.norm(a, ord=2) / len(a))
            l1 = np.linalg.norm(temp, ord=2)
            l2 = np.linalg.norm(paras, ord=2)
            cos = temp.dot(paras) / (l1 * l2)
            print('angle', np.arccos(cos))
            angle.append(np.arccos(cos))
            paras = temp

        train_loss.append(t_loss)
        train_acc.append(t_acc)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
    end = time.time()
    # print('successfully save weights, take {}s'.format(end - start))
    return end - start, np.array(train_loss), np.array(train_acc), np.array(val_loss), np.array(val_acc), np.array(
        norm), np.array(angle), np.array(temps)


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


def MNIST():
    use_cuda = torch.cuda.is_available()
    # statistic = {}
    # for key in net_type:
    #     statistic[key] = []  # time,train time, test time, test loss, test accuracy
    # ps = []
    temps = []
    # plt.figure(1)
    # plt.xlabel('epochs')
    # plt.ylabel('loss value')
    # plt.title('training loss vs validation loss')
    # plt.figure(2)
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy probability')
    # plt.title('training accuracy vs validation accuracy')
    epochs = net_epochs['MNIST']
    for j, key in enumerate(net_type):
        cost = torch.nn.CrossEntropyLoss()  # since pi from softmax function, this Loss is softmax Loss
        # temp = []
        model = LeNet(name='MNIST_net')

        # start = time.time()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        if key == 'M0':
            train_set, val_set, test_set = load.get_MNIST(random=False, p=None)
        if key == 'M1':
            p = M1(28)
            # ps.append(p)
            train_set, val_set, test_set = load.get_MNIST(random=True, p=p)
        if key == 'M2':
            p = M2(28)
            # ps.append(p)
            train_set, val_set, test_set = load.get_MNIST(random=True, p=p)

        # train_model(net=model,
        #             cost=cost,
        #             optimizer=optimizer,
        #             n_epochs=epochs,
        #             train_set=train_set,
        #             val_set=val_set,
        #             use_cuda=use_cuda)
        # plt.figure(3)
        # plt.xlabel('epochs')
        # plt.ylabel('norm value')
        # plt.title('difference value of norm {}'.format(key))
        # plt.figure(4)
        # plt.xlabel('epochs')
        # plt.ylabel('angle value')
        # plt.title('difference value of angle {}'.format(key))
        train_time, train_loss, train_acc, val_loss, val_acc, norm, angle, weight = train_model(net=model,
                                                                                                cost=cost,
                                                                                                optimizer=optimizer,
                                                                                                n_epochs=epochs,
                                                                                                train_set=train_set,
                                                                                                val_set=val_set,
                                                                                                use_cuda=use_cuda)
        temps.append(weight)
        # x = np.linspace(0, epochs, epochs)
        # plt.figure(1)
        # plt.plot(x, train_loss, 'g' + line[j], label='M' + str(j) + ' training loss')
        # plt.plot(x, val_loss, 'b' + line[j], label='M' + str(j) + ' validation loss')
        # plt.legend(loc='upper right')
        # plt.figure(2)
        # plt.plot(x, train_acc, 'g' + line[j], label='M' + str(j) + ' training accuracy')
        # plt.plot(x, val_acc, 'b' + line[j], label='M' + str(j) + ' validation accuracy')
        # plt.legend(loc='lower right')
        # plt.figure(3)
        # plt.plot(x, norm)
        # plt.savefig(pic_path + key + '_MNIST_norm.png')
        # plt.close()
        # plt.figure(4)
        # plt.plot(x, angle)
        # plt.savefig(pic_path + key + '_MNIST_angle.png')
        # plt.close()

        # end = time.time()
        # temp.append(start - end)
        # temp.append(train_time)
        # test_time, loss, acc = test_model(net=model,
        #                                   cost=cost,
        #                                   test_set=test_set,
        #                                   use_cuda=use_cuda)
        # temp.append(test_time)
        # temp.append(loss)
        # temp.append(acc)
        # temp.append(train_loss[-1])
        # temp.append(train_acc[-1])
        # temp.append(val_loss[-1])
        # temp.append(val_acc[-1])
        # statistic[key].append(temp)
    # plt.figure(1)
    # plt.savefig(pic_path + 'MNIST_loss.png')
    # plt.close()
    # plt.figure(2)
    # plt.savefig(pic_path + 'MNIST_acc.png')
    # plt.close()

    temp1 = temps[0]
    temp2 = temps[1]
    temp3 = temps[2]
    l0 = np.linalg.norm(temp1[0], ord=2)
    l1 = np.linalg.norm(temp1[-1],ord=2)
    cos01 = temp1[0].dot(temp1[-1]) / (l0 * l1)
    print(np.arccos(cos01))
    l0 = np.linalg.norm(temp2[0], ord=2)
    l1 = np.linalg.norm(temp2[-1], ord=2)
    cos01 = temp2[0].dot(temp2[-1]) / (l0 * l1)
    print(np.arccos(cos01))
    l0 = np.linalg.norm(temp3[0], ord=2)
    l1 = np.linalg.norm(temp3[-1], ord=2)
    cos01 = temp3[0].dot(temp3[-1]) / (l0 * l1)
    print(np.arccos(cos01))
    # l = len(temp1)
    # n_01 = []
    # n_02 = []
    # n_12 = []
    # a_01 = []
    # a_02 = []
    # a_12 = []
    # for i in range(l):
    #     c_01 = temp2[i] - temp1[i]
    #     c_02 = temp3[i] - temp1[i]
    #     c_12 = temp3[i] - temp2[i]
    #     n_01.append(np.linalg.norm(c_01, ord=2) / len(c_01))
    #     n_02.append(np.linalg.norm(c_02, ord=2) / len(c_02))
    #     n_12.append(np.linalg.norm(c_12, ord=2) / len(c_12))
    #     l0 = np.linalg.norm(temp1[i], ord=2)
    #     l1 = np.linalg.norm(temp2[i], ord=2)
    #     l2 = np.linalg.norm(temp3[i], ord=2)
    #     cos01 = temp1[i].dot(temp2[i]) / (l0 * l1)
    #     cos02 = temp1[i].dot(temp3[i]) / (l0 * l2)
    #     cos12 = temp2[i].dot(temp3[i]) / (l1 * l2)
    #     a_01.append(np.arccos(cos01))
    #     a_02.append(np.arccos(cos02))
    #     a_12.append(np.arccos(cos12))
    # x = np.linspace(0, epochs, epochs)
    # plt.figure(5)
    # plt.xlabel('epochs')
    # plt.ylabel('norm value')
    # plt.title('comparision of different weight')
    # plt.plot(x, np.array(n_01), label='M0 vs M1')
    # plt.plot(x, np.array(n_02), label='M0 vs M2')
    # plt.plot(x, np.array(n_12), label='M1 vs M2')
    # plt.legend(loc='upper right')
    # plt.savefig(pic_path + 'MNIST_comparision_norm.png')
    # plt.close()
    #
    # plt.figure(6)
    # plt.xlabel('epochs')
    # plt.ylabel('angle value')
    # plt.title('comparision of different weight')
    # plt.plot(x, np.array(a_01), label='M0 vs M1')
    # plt.plot(x, np.array(a_02), label='M0 vs M2')
    # plt.plot(x, np.array(a_12), label='M1 vs M2')
    # plt.legend(loc='upper right')
    # plt.savefig(pic_path + 'MNIST_comparision_angle.png')
    # plt.close()

    # with open("t_MNIST.txt", "w") as f:
    #     for key in statistic.keys():
    #         f.write('type is {} \n'.format(key))
    #         for item in statistic[key]:
    #             f.write(
    #                 'this process spends totally {}s, train spends {}s, test spends {}s, '
    #                 'test loss is {}, test accuracy is {}; train loss is {}, train accuracy is {}'
    #                 'validation loss is {}, validation accuracy is {}\n'.format(
    #                     item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]))


def CIFAR():
    use_cuda = torch.cuda.is_available()
    # statistic = {}
    # for key in net_type:
    #     statistic[key] = []  # time,train time, test time, test loss, test accuracy
    # ps = []
    temps = []
    # plt.figure(1)
    # plt.xlabel('epochs')
    # plt.ylabel('loss value')
    # plt.title('training loss vs validation loss')
    # plt.figure(2)
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy probability')
    # plt.title('training accuracy vs validation accuracy')
    epochs = net_epochs['CIFAR']
    for j, key in enumerate(net_type):
        cost = torch.nn.CrossEntropyLoss()  # since pi from softmax function, this Loss is softmax Loss
        # temp = []
        model = CIFAR_Net(name='CIFAR_net')

        # start = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
        if key == 'M0':
            train_set, val_set, test_set = load.get_CIFAR(random=False, p=None)
        if key == 'M1':
            p = M1(32)
            # ps.append(p)
            train_set, val_set, test_set = load.get_CIFAR(random=True, p=p)
        if key == 'M2':
            p = M2(32)
            # ps.append(p)
            train_set, val_set, test_set = load.get_CIFAR(random=True, p=p)

        # train_model(net=model,
        #             cost=cost,
        #             optimizer=optimizer,
        #             n_epochs=epochs,
        #             train_set=train_set,
        #             val_set=val_set,
        #             use_cuda=use_cuda)
        plt.figure(3)
        plt.xlabel('epochs')
        plt.ylabel('norm value')
        plt.title('difference value of norm {}'.format(key))
        plt.figure(4)
        plt.xlabel('epochs')
        plt.ylabel('angle value')
        plt.title('difference value of angle {}'.format(key))
        train_time, train_loss, train_acc, val_loss, val_acc, norm, angle, weight = train_model(net=model,
                                                                                                cost=cost,
                                                                                                optimizer=optimizer,
                                                                                                n_epochs=epochs,
                                                                                                train_set=train_set,
                                                                                                val_set=val_set,
                                                                                                use_cuda=use_cuda)
        temps.append(weight)
        # x = np.linspace(0, epochs, epochs)
        # plt.figure(1)
        # plt.plot(x, train_loss, 'g' + line[j], label='M' + str(j) + ' training loss')
        # plt.plot(x, val_loss, 'b' + line[j], label='M' + str(j) + ' validation loss')
        # plt.legend(loc='upper right')
        # plt.figure(2)
        # plt.plot(x, train_acc, 'g' + line[j], label='M' + str(j) + ' training accuracy')
        # plt.plot(x, val_acc, 'b' + line[j], label='M' + str(j) + ' validation accuracy')
        # plt.legend(loc='lower right')
        # plt.figure(3)
        # plt.plot(x, norm)
        # plt.savefig(pic_path + key + '_CIFAR_norm.png')
        # plt.close()
        # plt.figure(4)
        # plt.plot(x, angle)
        # plt.savefig(pic_path + key + '_CIFAR_angle.png')
        # plt.close()

        # end = time.time()
        # temp.append(start - end)
        # temp.append(train_time)
        # test_time, loss, acc = test_model(net=model,
        #                                   cost=cost,
        #                                   test_set=test_set,
        #                                   use_cuda=use_cuda)
        # temp.append(test_time)
        # temp.append(loss)
        # temp.append(acc)
        # temp.append(train_loss[-1])
        # temp.append(train_acc[-1])
        # temp.append(val_loss[-1])
        # temp.append(val_acc[-1])
        # statistic[key].append(temp)
    # plt.figure(1)
    # plt.savefig(pic_path + 'CIFAR_loss.png')
    # plt.close()
    # plt.figure(2)
    # plt.savefig(pic_path + 'CIFAR_acc.png')
    # plt.close()

    temp1 = temps[0]
    temp2 = temps[1]
    temp3 = temps[2]
    l0 = np.linalg.norm(temp1[0], ord=2)
    l1 = np.linalg.norm(temp1[-1], ord=2)
    cos01 = temp1[0].dot(temp1[-1]) / (l0 * l1)
    print(np.arccos(cos01))
    l0 = np.linalg.norm(temp2[0], ord=2)
    l1 = np.linalg.norm(temp2[-1], ord=2)
    cos01 = temp2[0].dot(temp2[-1]) / (l0 * l1)
    print(np.arccos(cos01))
    l0 = np.linalg.norm(temp3[0], ord=2)
    l1 = np.linalg.norm(temp3[-1], ord=2)
    cos01 = temp3[0].dot(temp3[-1]) / (l0 * l1)
    print(np.arccos(cos01))

    # temp1 = temps[0]
    # temp2 = temps[1]
    # temp3 = temps[2]
    # l = len(temp1)
    # n_01 = []
    # n_02 = []
    # n_12 = []
    # a_01 = []
    # a_02 = []
    # a_12 = []
    # for i in range(l):
    #     c_01 = temp2[i] - temp1[i]
    #     c_02 = temp3[i] - temp1[i]
    #     c_12 = temp3[i] - temp2[i]
    #     n_01.append(np.linalg.norm(c_01, ord=2) / len(c_01))
    #     n_02.append(np.linalg.norm(c_02, ord=2) / len(c_02))
    #     n_12.append(np.linalg.norm(c_12, ord=2) / len(c_12))
    #     l0 = np.linalg.norm(temp1[i], ord=2)
    #     l1 = np.linalg.norm(temp2[i], ord=2)
    #     l2 = np.linalg.norm(temp3[i], ord=2)
    #     cos01 = temp1[i].dot(temp2[i]) / (l0 * l1)
    #     cos02 = temp1[i].dot(temp3[i]) / (l0 * l2)
    #     cos12 = temp2[i].dot(temp3[i]) / (l1 * l2)
    #     a_01.append(np.arccos(cos01))
    #     a_02.append(np.arccos(cos02))
    #     a_12.append(np.arccos(cos12))
    # x = np.linspace(0, epochs, epochs)
    # plt.figure(5)
    # plt.xlabel('epochs')
    # plt.ylabel('norm value')
    # plt.title('comparision of different weight')
    # plt.plot(x, np.array(n_01), label='M0 vs M1')
    # plt.plot(x, np.array(n_02), label='M0 vs M2')
    # plt.plot(x, np.array(n_12), label='M1 vs M2')
    # plt.legend(loc='upper right')
    # plt.savefig(pic_path + 'CIFAR_comparision_norm.png')
    # plt.close()
    #
    # plt.figure(6)
    # plt.xlabel('epochs')
    # plt.ylabel('angle value')
    # plt.title('comparision of different weight')
    # plt.plot(x, np.array(a_01), label='M0 vs M1')
    # plt.plot(x, np.array(a_02), label='M0 vs M2')
    # plt.plot(x, np.array(a_12), label='M1 vs M2')
    # plt.legend(loc='upper right')
    # plt.savefig(pic_path + 'CIFAR_comparision_angle.png')
    # plt.close()

    # with open("t_MNIST.txt", "w") as f:
    #     for key in statistic.keys():
    #         f.write('type is {} \n'.format(key))
    #         for item in statistic[key]:
    #             f.write(
    #                 'this process spends totally {}s, train spends {}s, test spends {}s, '
    #                 'test loss is {}, test accuracy is {}; train loss is {}, train accuracy is {}'
    #                 'validation loss is {}, validation accuracy is {}\n'.format(
    #                     item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]))


def C_MNIST(times=3):
    use_cuda = torch.cuda.is_available()

    epochs = net_epochs['MNIST']
    for j, key in enumerate(net_type):
        temps = []
        for i in range(times):
            cost = torch.nn.CrossEntropyLoss()  # since pi from softmax function, this Loss is softmax Loss
            # temp = []
            model = LeNet(name='MNIST_net')

            # start = time.time()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            if key == 'M0':
                train_set, val_set, test_set = load.get_MNIST(random=False, p=None)
            if key == 'M1':
                p = M1(28)
                # ps.append(p)
                train_set, val_set, test_set = load.get_MNIST(random=True, p=p)
            if key == 'M2':
                p = M2(28)
                # ps.append(p)
                train_set, val_set, test_set = load.get_MNIST(random=True, p=p)

            train_time, train_loss, train_acc, val_loss, val_acc, norm, angle, weight = train_model(net=model,
                                                                                                    cost=cost,
                                                                                                    optimizer=optimizer,
                                                                                                    n_epochs=epochs,
                                                                                                    train_set=train_set,
                                                                                                    val_set=val_set,
                                                                                                    use_cuda=use_cuda)
            temps.append(weight)
        temp1 = temps[0]
        temp2 = temps[1]
        temp3 = temps[2]
        l = len(temp1)
        n_01 = []
        n_02 = []
        n_12 = []
        a_01 = []
        a_02 = []
        a_12 = []
        for i in range(l):
            c_01 = temp2[i] - temp1[i]
            c_02 = temp3[i] - temp1[i]
            c_12 = temp3[i] - temp2[i]
            n_01.append(np.linalg.norm(c_01, ord=2) / len(c_01))
            n_02.append(np.linalg.norm(c_02, ord=2) / len(c_02))
            n_12.append(np.linalg.norm(c_12, ord=2) / len(c_12))
            l0 = np.linalg.norm(temp1[i], ord=2)
            l1 = np.linalg.norm(temp2[i], ord=2)
            l2 = np.linalg.norm(temp3[i], ord=2)
            cos01 = temp1[i].dot(temp2[i]) / (l0 * l1)
            cos02 = temp1[i].dot(temp3[i]) / (l0 * l2)
            cos12 = temp2[i].dot(temp3[i]) / (l1 * l2)
            a_01.append(np.arccos(cos01))
            a_02.append(np.arccos(cos02))
            a_12.append(np.arccos(cos12))
        x = np.linspace(0, epochs, epochs)
        plt.figure(1)
        plt.xlabel('epochs')
        plt.ylabel('norm value')
        plt.title('comparision of different weight')
        plt.plot(x, np.array(n_01), label='0 vs 1')
        plt.plot(x, np.array(n_02), label='0 vs 2')
        plt.plot(x, np.array(n_12), label='1 vs 2')
        plt.legend(loc='upper right')
        plt.savefig(pic_path + 'MNIST_{}_3_norm.png'.format(key))
        plt.close()

        plt.figure(2)
        plt.xlabel('epochs')
        plt.ylabel('angle value')
        plt.title('comparision of different weight')
        plt.plot(x, np.array(a_01), label='0 vs 1')
        plt.plot(x, np.array(a_02), label='0 vs 2')
        plt.plot(x, np.array(a_12), label='1 vs 2')
        plt.legend(loc='upper right')
        plt.savefig(pic_path + 'MNIST_{}_3_angle.png'.format(key))
        plt.close()


def C_CIFAR(times = 3):
    use_cuda = torch.cuda.is_available()

    epochs = net_epochs['CI']
    for j, key in enumerate(net_type):
        temps = []
        for i in range(times):
            cost = torch.nn.CrossEntropyLoss()  # since pi from softmax function, this Loss is softmax Loss
            # temp = []
            model = CIFAR_Net(name='CIFAR_net')

            # start = time.time()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
            if key == 'M0':
                train_set, val_set, test_set = load.get_CIFAR(random=False, p=None)
            if key == 'M1':
                p = M1(32)
                # ps.append(p)
                train_set, val_set, test_set = load.get_CIFAR(random=True, p=p)
            if key == 'M2':
                p = M2(32)
                # ps.append(p)
                train_set, val_set, test_set = load.get_CIFAR(random=True, p=p)

            train_time, train_loss, train_acc, val_loss, val_acc, norm, angle, weight = train_model(net=model,
                                                                                                    cost=cost,
                                                                                                    optimizer=optimizer,
                                                                                                    n_epochs=epochs,
                                                                                                    train_set=train_set,
                                                                                                    val_set=val_set,
                                                                                                    use_cuda=use_cuda)
            temps.append(weight)
        temp1 = temps[0]
        temp2 = temps[1]
        temp3 = temps[2]
        l = len(temp1)
        n_01 = []
        n_02 = []
        n_12 = []
        a_01 = []
        a_02 = []
        a_12 = []
        for i in range(l):
            c_01 = temp2[i] - temp1[i]
            c_02 = temp3[i] - temp1[i]
            c_12 = temp3[i] - temp2[i]
            n_01.append(np.linalg.norm(c_01, ord=2) / len(c_01))
            n_02.append(np.linalg.norm(c_02, ord=2) / len(c_02))
            n_12.append(np.linalg.norm(c_12, ord=2) / len(c_12))
            l0 = np.linalg.norm(temp1[i], ord=2)
            l1 = np.linalg.norm(temp2[i], ord=2)
            l2 = np.linalg.norm(temp3[i], ord=2)
            cos01 = temp1[i].dot(temp2[i]) / (l0 * l1)
            cos02 = temp1[i].dot(temp3[i]) / (l0 * l2)
            cos12 = temp2[i].dot(temp3[i]) / (l1 * l2)
            a_01.append(np.arccos(cos01))
            a_02.append(np.arccos(cos02))
            a_12.append(np.arccos(cos12))
        x = np.linspace(0, epochs, epochs)
        plt.figure(1)
        plt.xlabel('epochs')
        plt.ylabel('norm value')
        plt.title('comparision of different weight')
        plt.plot(x, np.array(n_01), label='0 vs 1')
        plt.plot(x, np.array(n_02), label='0 vs 2')
        plt.plot(x, np.array(n_12), label='1 vs 2')
        plt.legend(loc='upper right')
        plt.savefig(pic_path + 'CIFAR_{}_3_norm.png'.format(key))
        plt.close()

        plt.figure(2)
        plt.xlabel('epochs')
        plt.ylabel('angle value')
        plt.title('comparision of different weight')
        plt.plot(x, np.array(a_01), label='0 vs 1')
        plt.plot(x, np.array(a_02), label='0 vs 2')
        plt.plot(x, np.array(a_12), label='1 vs 2')
        plt.legend(loc='upper right')
        plt.savefig(pic_path + 'CIFAR_{}_3_angle.png'.format(key))
        plt.close()


# CIFAR_net = ['conv', 'ReLu', 'pooling', 'fc']
#
#
# def CIFAR(layer=1):
#     model = nn.Sequential()
#     for i in range(layer):
#         if CIFAR_net[i] == 'conv':
#             conv = nn.Conv2d(3, 128, 13)
#             model.add_module('conv', conv)
#         elif CIFAR_net[i] == 'ReLu':
#             relu = nn.ReLU()
#             model.add_module('ReLu',relu)
#         elif CIFAR_net[i] == 'pooling':
#             pool = nn.AvgPool2d(3,2)
#             model.add_module('pooling',pool)
#         elif CIFAR_net[i] == 'fc':
#             fc = nn.


def main():
    CIFAR()
    # MNIST()
    # C_MNIST()
    # C_CIFAR()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
