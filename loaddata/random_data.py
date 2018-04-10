import torch


def sample_noise(size, dim, mean, std):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """

    torch.manual_seed(1)
    data = torch.ones(size, dim * dim)
    temp = torch.normal(mean * data, std)
    norm = torch.max(torch.abs(temp))
    return temp/norm


# @parameter
# channel:the channel of input images
# size: the amount of elements in a batch of random data
# dim: the size of random image is dim*dim

def get_random_data(channel, size, dim, mean, std):
    raw_data = sample_noise(size, dim, mean, std)
    return raw_data.view(raw_data.size(0), channel, dim, dim)
