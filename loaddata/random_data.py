import torch


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    temp = torch.rand(batch_size, dim * dim) + torch.rand(batch_size, dim * dim) * (-1)

    return temp


def get_random_data(data_size, channel, batch_size, dim):
    time = data_size // batch_size + 1
    temp = []
    for i in range(time):
        if i == time - 1:
            temp.append(sample_noise(data_size - (time - 1) * batch_size, dim))
        else:
            temp.append(sample_noise(batch_size, dim))

    raw_data = torch.cat(tuple(temp), 0)
    return raw_data.view(raw_data.size(0), channel, dim, dim)
