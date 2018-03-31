import torch


def sample_noise(size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    temp = torch.rand(size, dim * dim) + torch.rand(size, dim * dim) * (-1)

    return temp

def sample_constant(constant,size,dim):
    temp = constant*torch.ones(size, dim * dim)
    return temp

# @parameter
# channel:the channel of input images
# size: the amount of elements in a batch of random data
# dim: the size of random image is dim*dim

def get_random_data(channel, size, dim):
    raw_data = sample_noise(size, dim)
    return raw_data.view(raw_data.size(0), channel, dim, dim)


def get_constant_value(constant,channel,size,dim):
    raw_data = sample_constant(constant,size,dim)
    return raw_data.view(raw_data.size(0), channel, dim, dim)

