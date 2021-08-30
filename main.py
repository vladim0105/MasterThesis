import torch
import torchvision.models as models
from torchsummary import summary
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import vis_utils
import model as m
from FrameDataset import FrameDataset
import numpy as np
if __name__ == '__main__':

    dataset = FrameDataset("video_1.npy")
    dataset.create_anomalies(10)
    anomalies = []
    print("aaaa")
    htm_layer = m.HTMLayer(shape=(512,), columnDimensions=(512,))
    for i in range(len(dataset)):
        sample = dataset[i]
        data = torch.ones(size=(512,))
        if i > 100:
            data = torch.ones(size=(512,))
        pred, anom = htm_layer(data)
        print(data.sum())
        print(pred.sum())
        print(anom)
        anomalies.append(anom)
    print(anomalies)
    plt.plot(dataset.anomalies, color="red", label="anomalies")
    plt.plot(anomalies, color="blue", label="htm anomalies")
    plt.legend()
    plt.show()
