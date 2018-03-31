from torchvision import transforms
from loaddata.ran_Datasets import *

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 128


def get_MNIST(random=False):
    data_train = ran_MNIST(root="./data/MINST",
                           transform=transforms,
                           train=True,
                           download=True,
                           random=random)
    data_test = ran_MNIST(root="./data/MINST",
                          transform=transforms,
                          download=True,
                          train=False,
                          random=random)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    return data_loader_train, data_loader_test


def get_CIFAR(random=False):
    data_train = ran_CIFAR(root="./data/CIFAR",
                           transform=transforms,
                           train=True,
                           download=True,
                           random=random)

    data_test = ran_CIFAR(root="./data/CIFAR",
                          transform=transforms,
                          train=False,
                          random=random)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)

    return data_loader_train, data_loader_test
