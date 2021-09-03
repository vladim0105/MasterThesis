import matplotlib.image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import torch

class FrameDataset(Dataset):
    def __init__(self, file):
        self.dataset = np.load(file, allow_pickle=True).item()
        self.frames = self.dataset["frames"]
        self.anomalies = np.zeros(len(self.frames)).astype(np.bool)

    def create_anomalies(self, n, length, reverse=False):
        direction=-1 if reverse else 1
        # TODO make sure no overlap happens
        for i in range(n):
            indices = np.random.randint(0, len(self.frames), size=2)
            for j in range(length):
                try:
                    self.frames[indices[0]+j] = (self.frames[indices[1]+direction*j])
                    self.anomalies[indices[0]+j] = True
                except:
                    pass
        self.dataset["frames"] = self.frames

    def __getitem__(self, index):
        frame = self.frames[index]
        if index==0:
            matplotlib.image.imsave("out.png", frame, cmap="gray")
        return torch.Tensor(frame)

    def __len__(self):
        return len(self.frames)
