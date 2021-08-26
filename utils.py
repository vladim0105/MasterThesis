import torch
from htm.bindings.sdr import SDR


def tensor_to_sdr(tensor: torch.Tensor) -> SDR:
    sdr = SDR(dimensions=tensor.shape)
    sdr.dense = tensor.tolist()

    return sdr


def sdr_to_tensor(sdr: SDR) -> torch.Tensor:
    tensor = torch.Tensor(sdr.dense)
    return tensor
