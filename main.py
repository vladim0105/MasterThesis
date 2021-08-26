import torch

import utils

if __name__ == '__main__':
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    test = torch.Tensor(data)
    print(utils.tensor_to_sdr(test).dense)
