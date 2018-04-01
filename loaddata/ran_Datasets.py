import torch
from torchvision import datasets
import numpy as np

'''
    this class is inherited form datasets.MNIST, what I did is shuffling original order of every data in MNIST,
    including line's and row's

'''


class ran_MNIST(datasets.MNIST):
    def __init__(self, root, transform=None, target_transform=None, download=False, random=False):
        super(ran_MNIST, self).__init__(root, True, transform, target_transform, download)
        super(ran_MNIST, self).__init__(root, False, transform, target_transform, download)
        if random:
            temp1 = np.copy(self.train_data.numpy())
            size1 = temp1.shape[0]
            temp2 = np.copy(self.test_data.numpy())
            temp = np.concatenate((temp1, temp2), axis=0)
            temp = self.random_resize(temp)
            self.train_data = torch.ByteTensor(temp[:size1])
            self.test_data = torch.ByteTensor(temp[size1:])

    def set_mode(self, train=True):
        if train:
            self.train = True
        else:
            self.train = False

    def random_resize(self, datas):
        size = datas.shape[0]
        temp = np.hstack(tuple(datas))
        np.random.shuffle(temp)
        temp = np.hsplit(temp, size)
        temp = np.vstack(tuple(temp))
        np.random.shuffle(temp.T)
        temps = np.vsplit(temp, size)
        return np.array(temps)
        # temps = []
        # for data in datas:
        #     temp = np.copy(data.numpy())
        #     np.random.shuffle(temp)
        #     np.random.shuffle(temp.T)
        #     temps.append(temp.T)
        # temps = np.array(temps)
        # return torch.ByteTensor(temps)


'''
   like ran_MNIST
'''


class ran_CIFAR(datasets.CIFAR10):
    def __init__(self, root, transform=None, target_transform=None, download=False, random=False):
        super(ran_CIFAR, self).__init__(root, True, transform, target_transform, download)
        super(ran_CIFAR, self).__init__(root, False, transform, target_transform, download)

        if random:
            self.train_data = self.train_data.transpose((0, 3, 1, 2))
            self.test_data = self.test_data.transpose((0, 3, 1, 2))
            temp1 = np.copy(self.train_data)
            size1 = temp1.shape[0]
            temp2 = np.copy(self.test_data)
            temp = np.concatenate((temp1, temp2), axis=0)
            temp = self.random_resize(temp)

            self.train_data = temp[:size1]
            self.test_data = temp[size1:]

            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def set_mode(self, train=True):
        if train:
            self.train = True
        else:
            self.train = False

    def random_resize(self, datas):
        temps = []
        size = datas.shape[0]
        channel = datas.shape[1]
        for data in datas:
            for item in data:
                temps.append(item)

        temps = np.hstack(tuple(temps))
        np.random.shuffle(temps)
        temps = np.hsplit(temps, size * channel)
        temps = np.vstack(tuple(temps))
        np.random.shuffle(temps.T)
        temps = np.vsplit(temps, size * channel)

        result = []
        temp = []
        for i, item in enumerate(temps):
            if i % 3 == 0 and not i == 0:
                temp = []
            temp.append(item)
            if i % 3 == 2:
                result.append(np.array(temp))
        print(np.array(result).shape)
        return np.array(result)

        # temps = []
        # for data in datas:
        #     rtemp = []
        #     for item in data:
        #         rtemp.append(np.copy(item))
        #     row = np.hstack(tuple(rtemp))
        #     np.random.shuffle(row)
        #     rtemp = np.hsplit(row,3)
        #     ltemp = []
        #     for item in rtemp:
        #         ltemp.append(item.T)
        #     line = np.hstack(tuple(ltemp))
        #     np.random.shuffle(line)
        #     ltemp = np.hsplit(line,3)
        #     temp = []
        #     for item in ltemp:
        #         temp.append(item.T)
        #     temps.append(temp)
        # temps = np.array(temps)
        # return temps
