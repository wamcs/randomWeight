import torch
from torchvision import datasets
import numpy as np

'''
    this class is inherited form datasets.MNIST, what I did is shuffling original order of every data in MNIST,
    including line's and row's

'''
base1 = np.random.permutation(range(28*28))
base2 = np.random.permutation(range(32*32))


class ran_MNIST(datasets.MNIST):
    def __init__(self, root, transform=None, target_transform=None, download=False, random=False):
        super(ran_MNIST, self).__init__(root, True, transform, target_transform, download)
        super(ran_MNIST, self).__init__(root, False, transform, target_transform, download)
        if random:
            temp1 = np.copy(self.train_data.numpy())
            print(temp1[0])
            size1 = temp1.shape[0]
            temp2 = np.copy(self.test_data.numpy())
            temp = np.concatenate((temp1, temp2), axis=0)
            temp = self.random_resize(temp)
            self.train_data = torch.from_numpy(temp[:size1])
            self.test_data = torch.from_numpy(temp[size1:])

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


class all_ran_MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, random=False):
        super(all_ran_MNIST, self).__init__(root, train, transform, target_transform, download)
        if random:
            if train:
                self.train_data = self.random_resize(self.train_data)
            else:
                self.test_data = self.random_resize(self.test_data)

    def permute_mnist(self, x, n, p):
        """ permute mnist images

        Args:
             x: input tensor with shape: [n, 784]
             n: number of images
             p: random permute matrix

        Returns:
            x_permute: output tensor with permute
        """
        x_permute = np.zeros([n, 784])
        for i in range(n):
            for j in range(784):
                x_permute[i][j] = x[i][p[j]]
        return x_permute

    def random_resize(self, datas):
        temp = np.copy(datas.numpy())
        size = temp.shape[0]
        temp = temp.reshape((size, -1))
        out = self.permute_mnist(temp, size, base1)
        out = out.reshape((size, 28, 28))
        return torch.from_numpy(out)


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

class all_ran_CIFAR(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, random=False):
        super(all_ran_CIFAR, self).__init__(root, train, transform, target_transform, download)
        if random:
            if train:
                self.train_data = self.train_data.transpose((0, 3, 1, 2))
                self.train_data = self.random_resize(self.train_data)
                self.train_data = self.train_data.transpose((0, 2, 3, 1))
            else:
                self.test_data = self.test_data.transpose((0, 3, 1, 2))
                self.test_data = self.random_resize(self.test_data)
                self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def permute_mnist(self, x, n, p):
        """ permute mnist images

        Args:
             x: input tensor with shape: [n, 784]
             n: number of images
             p: random permute matrix

        Returns:
            x_permute: output tensor with permute
        """
        x_permute = np.zeros([n, 3, 32*32])
        for i in range(n):
            for j in range(3):
                for k in range(32*32):
                    x_permute[i][j][k] = x[i][j][p[k]]
        return x_permute

    def random_resize(self, datas):
        temp = np.copy(datas)
        size = temp.shape[0]
        temp = temp.reshape((size,3, -1))
        out = self.permute_mnist(temp, size, base2)
        out = out.reshape((size, 3, 32, 32))
        return out

