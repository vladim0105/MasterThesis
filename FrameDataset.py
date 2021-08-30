import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import torch

class FrameDataset(Dataset):
    def __init__(self, file):
        dataset = np.load(file, allow_pickle=True).item()
        self.frames = dataset["frames"]
        self.anomalies = np.zeros(len(self.frames)).astype(np.bool)

    def create_anomalies(self, n):
        # TODO make sure no overlap happens
        for i in range(n):
            indices = np.random.randint(0, len(self.frames), size=2)
            self.frames[indices[0]] = self.frames[indices[1]]
            self.anomalies[indices[0]] = True

    def __getitem__(self, index):
        frame = self.frames[index]
        frame = np.where(frame != 0, 1, 0)
        return torch.Tensor(frame)

    def __len__(self):
        return len(self.frames)
