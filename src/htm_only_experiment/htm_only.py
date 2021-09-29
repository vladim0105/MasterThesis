import torch
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import progressbar
import dataset_to_video
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import vis_utils
import model as m
from FrameDataset import FrameDataset
import numpy as np
from datetime import datetime
if __name__ == '__main__':
    np.random.seed(2)
    dataset = FrameDataset("video_1.npy")

    anomalies = []
    anomalies_likelihood=[]
    preds = []
    cnn_layer = m.CNNLayer()
    htm_layer = m.HTMLayer(shape=(140, 120), columnDimensions=(50, 50), seed=1)
    dataset_to_video.dataset_to_video(dataset.dataset, "test.avi")
    # First, expose the HTM network to the video several times
    n=5
    with progressbar.ProgressBar(max_value=n * len(dataset)) as bar:
        for j in range(n):
            for i in range(len(dataset)):
                sample = dataset[i]
                htm_layer(sample, True)
                bar.update(bar.value+1)
    dataset_to_video.dataset_to_video(dataset.dataset, "test.avi")
    #Then expose it to the anomalies
    dataset = FrameDataset("video_2.npy")
    dataset.create_anomalies(5, 20, True)
    dataset_to_video.dataset_to_video(dataset.dataset, "test2.avi")
    for i in range(len(dataset)):
        sample = dataset[i]
        pred, anom = htm_layer(sample, True)
        preds.append(pred)
        anomalies.append(anom)

    plt.plot(dataset.anomalies, color="red", label="anomalies")
    plt.plot(anomalies, color="blue", label="htm anomalies")
    plt.ylim([0, 1.2])
    plt.xlabel("Frame #")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.savefig(f"plots/{datetime.now()}.png")
    plt.show()
