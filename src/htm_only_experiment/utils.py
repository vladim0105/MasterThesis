import torch
from htm.bindings.sdr import SDR
import torch.nn as nn
import numpy as np
from htm.encoders.rdse import RDSE


def tensor_to_sdr(tensor: torch.Tensor) -> SDR:
    sdr = SDR(dimensions=tensor.shape)
    sdr.dense = tensor.tolist()

    return sdr


def sdr_to_tensor(sdr: SDR) -> torch.Tensor:
    tensor = torch.Tensor(sdr.dense)
    return tensor


def float_array_to_sdr(array: np.ndarray, encoder: RDSE) -> SDR:
    flat = array.flatten()
    encodings = []
    for val in np.nditer(flat):
        encodings.append(encoder.encode(val))

    sdr = SDR(dimensions=(encodings[0].dimensions[0] * len(encodings),))

    sdr.concatenate(encodings)

    return sdr


import sys
def log(o: object, file: str):
    print(str(o))
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(file, 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(str(o))
        sys.stdout = original_stdout  # Reset the standard output to its original value
