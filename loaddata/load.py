from torchvision import transforms
from loaddata.ran_Datasets import *

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
vgg_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
batch_size = 300


def get_MNIST(random, p):
    train_datas = MNIST(root="./data/MNIST",
                        p=p,
                        type='train',
                        transform=transform,
                        download=True,
                        random=random)

    val_datas = MNIST(root="./data/MNIST",
                      p=p,
                      type='val',
                      transform=transform,
                      download=True,
                      random=random)

    test_datas = MNIST(root="./data/MNIST",
                       p=p,
                       type='test',
                       transform=transform,
                       download=True,
                       random=random)

    data_loader_train = torch.utils.data.DataLoader(dataset=train_datas,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=1)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_datas,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(dataset=test_datas,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    return data_loader_train, data_loader_val, data_loader_test


def get_CIFAR(random, p):
    train_datas = CIFAR(root="./data/CIFAR",
                        p=p,
                        type='train',
                        transform=transform,
                        download=True,
                        random=random)

    val_datas = CIFAR(root="./data/CIFAR",
                      p=p,
                      type='val',
                      transform=transform,
                      download=True,
                      random=random)

    test_datas = CIFAR(root="./data/CIFAR",
                       p=p,
                       type='test',
                       transform=transform,
                       download=True,
                       random=random)

    data_loader_train = torch.utils.data.DataLoader(dataset=train_datas,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=1)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_datas,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(dataset=test_datas,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    return data_loader_train, data_loader_val, data_loader_test


def get_imageNet(random, p, category):
    train_datas = ImageNet(root='/home/data/data/ImageNet_ilsvrc2012_2014/train/',
                           p=p,
                           category=category,
                           type='train',
                           transform=vgg_transform,
                           random=random)

    val_datas = ImageNet(root='/home/data/data/ImageNet_ilsvrc2012_2014/train/',
                         p=p,
                         category=category,
                         type='val',
                         transform=vgg_transform,
                         random=random)

    test_datas = ImageNet(root="/home/data/data/ImageNet_ilsvrc2012_2014/train/",
                          p=p,
                          category=category,
                          type='test',
                          transform=vgg_transform,
                          random=random)

    data_loader_train = torch.utils.data.DataLoader(dataset=train_datas,
                                                    batch_size=50,
                                                    shuffle=True,
                                                    num_workers=1)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_datas,
                                                  batch_size=50,
                                                  shuffle=True,
                                                  num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(dataset=test_datas,
                                                   batch_size=50,
                                                   shuffle=True,
                                                   num_workers=1)
    return data_loader_train, data_loader_val, data_loader_test
