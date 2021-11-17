import torch
import numpy as np
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


def label_rgb_mask(mask_rgb: torch.Tensor, mapping):
    mask = torch.zeros(size=(mask_rgb.shape[1], mask_rgb.shape[2]))
    # Convert from RGB mask to class mask
    for k in mapping:
        # Get all indices for current class
        idx = (mask_rgb == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        idx = (idx.sum(0) == 3)
        mask[idx] = mapping[k]
    return mask


def label_rgb_mask_numpy(mask_rgb, mask_mapping):
    labels = np.zeros((mask_rgb.shape[:2]), dtype=np.int)
    for k in mask_mapping:
        labels[(mask_rgb == k).all(axis=2)] = mask_mapping[k]
    return labels


def unpack_mask(mask: torch.Tensor, num_classes, mapping):
    assert len(mask.shape) == 4, "Wrong input dimension!"
    mask_out = torch.zeros(size=(mask.shape[0], num_classes, mask.shape[2], mask.shape[3]))
    # Convert from RGB mask to class mask
    for b in range(mask_out.shape[0]):
        for k in mapping.values():
            # Get all indices for current class
            idx = (mask[b] == k).squeeze()
            mask_out[b, k, idx] = 1

    return mask_out
