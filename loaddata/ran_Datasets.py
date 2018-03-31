import torch
from torchvision import datasets
import numpy as np


'''
    this class is inherited form datasets.MNIST, what I did is shuffling original order of every data in MNIST,
    including line's and row's

'''
class ran_MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, random=False):
        super(ran_MNIST, self).__init__(root, train, transform, target_transform, download)
        if random:
            if train:
                self.random_resize(self.train_data)
            else:
                self.random_resize(self.test_data)

    def random_resize(self, datas):
        temps = []
        for data in datas:
            temp = data.numpy()
            np.random.shuffle(temp)
            np.random.shuffle(temp.T)
            temps.append(temp.T)
        temps = np.array(temps)
        return torch.ByteTensor(temps)

'''
   like ran_MNIST
'''
class ran_CIFAR(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, random=False):
        super(ran_CIFAR, self).__init__(root, train, transform, target_transform, download)

        if random:
            if train:
                self.train_data = self.train_data.transpose((0, 3, 1, 2))
                self.train_data = self.random_resize(self.train_data)
                self.train_data = self.train_data.transpose((0, 2, 3, 1))
            else:
                self.test_data = self.test_data.transpose((0, 3, 1, 2))
                self.test_data = self.random_resize(self.test_data)
                self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def random_resize(self,datas):
        temps = []
        for data in datas:
            rtemp = []
            for item in data:
                rtemp.append(item)
            row = np.hstack(tuple(rtemp))
            np.random.shuffle(row)
            rtemp = np.hsplit(row,3)
            ltemp = []
            for item in rtemp:
                ltemp.append(item.T)
            line = np.hstack(tuple(ltemp))
            np.random.shuffle(line)
            ltemp = np.hsplit(line,3)
            temp = []
            for item in ltemp:
                temp.append(item.T)
            temps.append(temp)
        temps = np.array(temps)
        return temps



