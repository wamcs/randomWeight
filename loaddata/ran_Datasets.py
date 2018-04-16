import torch
from torchvision import datasets
import numpy as np
from PIL import Image
import os
import os.path
import torch.utils.data as data

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

'''
    this class is inherited form datasets.MNIST, what I did is shuffling original order of every data in MNIST,
    including line's and row's

'''


class MNIST(datasets.MNIST):
    def __init__(self, root, p, train=True, transform=None, target_transform=None, download=False, random=False):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)
        self.p = p
        if random:
            if train:
                self.train_data = self.random_resize(self.train_data)
            else:
                self.test_data = self.random_resize(self.test_data)

    def permute_mnist(self, x, n):
        x_permute = np.zeros([n, 784])
        for i in range(n):
            for j in range(784):
                x_permute[i][j] = x[i][self.p[j]]
        return x_permute

    def random_resize(self, datas):
        temp = np.copy(datas.numpy())
        size = temp.shape[0]
        temp = temp.reshape((size, -1))
        out = self.permute_mnist(temp, size)
        out = out.reshape((size, 28, 28))
        return torch.from_numpy(out)


'''
like above
'''


class CIFAR(datasets.CIFAR10):
    def __init__(self, root, p, train=True, transform=None, target_transform=None, download=False, random=False):
        super(CIFAR, self).__init__(root, train, transform, target_transform, download)
        self.p = p
        if random:
            if train:
                self.train_data = self.train_data.transpose((0, 3, 1, 2))
                self.train_data = self.random_resize(self.train_data)
                self.train_data = self.train_data.transpose((0, 2, 3, 1))
            else:
                self.test_data = self.test_data.transpose((0, 3, 1, 2))
                self.test_data = self.random_resize(self.test_data)
                self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def permute_cifar(self, x, n):
        x_permute = np.zeros([n, 3, 32 * 32]).astype(dtype=np.uint8)
        for i in range(n):
            for j in range(3):
                for k in range(32 * 32):
                    x_permute[i][j][k] = x[i][j][self.p[k]]
        return x_permute

    def random_resize(self, datas):
        temp = np.copy(datas)
        size = temp.shape[0]
        temp = temp.reshape((size, 3, -1))
        out = self.permute_cifar(temp, size)
        out = out.reshape((size, 3, 32, 32))
        return out


'''
modify ImageLoader to get a loader which can generate random data which like above mentioned
'''


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


def make_dataset(dir, class_to_idx, catagory, train):
    images = []
    dir = os.path.expanduser(dir)
    for i, target in enumerate(sorted(os.listdir(dir))):
        if i >= catagory:
            break
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            l = len(fnames)
            board = l // 10
            if train:
                fnames = fnames[:-board]
            else:
                fnames = fnames[-board:]
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class ImageNet(data.Dataset):
    def __init__(self, root, p, category, train=True, transform=None, target_transform=None, random=False,
                 loader=default_loader):
        class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, category, train)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.root = root
        self.p = p
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.random = random

    def permute_Image(self, x):
        x_permute = np.zeros(224 * 224)
        for i in range(224 * 224):
            x_permute[i] = x[self.p[i]]
        return x_permute

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            if self.random:
                temp = img.numpy()
                temp = temp.reshape((3, -1))
                temps = []
                for i in range(3):
                    temps.append(self.permute_Image(temp[i]))
                temps = np.array(temps)
                temps = temps.reshape((3, 224, 224))
                img = torch.from_numpy(temps).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
