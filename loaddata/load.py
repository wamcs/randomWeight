from torchvision import transforms
from loaddata.ran_Datasets import *

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 300


def get_MNIST(random=False):
    datas = ran_MNIST(root="./data/MINST",
                      transform=transforms,
                      download=True,
                      random=random)

    return datas


def get_all_ran_MNIST(random):
    train_datas = all_ran_MNIST(root="./data/MINST",
                                train=True,
                                transform=transforms,
                                download=True,
                                random=random)

    test_datas = all_ran_MNIST(root="./data/MINST",
                               train=False,
                               transform=transforms,
                               download=True,
                               random=random)

    data_loader_train = torch.utils.data.DataLoader(dataset=train_datas,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(dataset=test_datas,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    return data_loader_train, data_loader_test


def get_CIFAR(random=False):
    datas = ran_CIFAR(root="./data/CIFAR",
                      transform=transforms,
                      download=True,
                      random=random)
    return datas

def get_all_ran_CIFAR(random):
    train_datas = all_ran_CIFAR(root="./data/CIFAR",
                                train=True,
                                transform=transforms,
                                download=True,
                                random=random)

    test_datas = all_ran_CIFAR(root="./data/CIFAR",
                               train=False,
                               transform=transforms,
                               download=True,
                               random=random)

    data_loader_train = torch.utils.data.DataLoader(dataset=train_datas,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(dataset=test_datas,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    return data_loader_train, data_loader_test
