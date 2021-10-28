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


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch