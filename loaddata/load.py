import torch
from torchvision import datasets, transforms

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 128


def get_MNIST():
    data_train = datasets.MNIST(root="./data/MINST",
                                transform=transforms,
                                train=True,
                                download=True)
    data_test = datasets.MNIST(root="./data/MINST",
                               transform=transforms,
                               download=True,
                               train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    return data_loader_train, data_loader_test


def get_CIFAR():
    data_train = datasets.CIFAR10(root="./data/CIFAR",
                                  transform=transforms,
                                  train=True,
                                  download=True)

    data_test = datasets.CIFAR10(root="./data/CIFAR",
                                 transform=transforms,
                                 train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=2)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

    return data_loader_train, data_loader_test
