import glob
import json
import os
import pickle
import typing
from pathlib import Path

import numpy as np
import progressbar
import torch
import torchvision.models.resnet
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from PIL import Image
from torchvision.transforms import transforms


class ImageReconstructionDataset(Dataset):
    def __init__(self, path, extension, device):
        self.files = glob.glob(f"{path}/*.{extension}")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = device

    def __getitem__(self, index) -> T_co:
        pil_img = Image.open(self.files[index]).convert('L')
        img = self.transform(pil_img).to(self.device)
        return img, img

    def __len__(self):
        return len(self.files)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, stride=(2, 2), out_channels=3, kernel_size=(5, 5), padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, stride=(2, 2), out_channels=6, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, stride=(2, 2), out_channels=12, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.upconv1 = nn.ConvTranspose2d(in_channels=12, stride=(2, 2), out_channels=6, kernel_size=(2, 2))
        self.upconv2 = nn.ConvTranspose2d(in_channels=6, stride=(2, 2), out_channels=3, kernel_size=(2, 2))
        self.upconv3 = nn.ConvTranspose2d(in_channels=3, stride=(2, 2), out_channels=1, kernel_size=(2, 2))

    def forward(self, x):
        x = self.upconv1(x)
        x = self.relu(x)
        x = self.upconv2(x)
        x = self.relu(x)
        x = self.upconv3(x)
        x = self.relu(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        latent_rep = self.encoder(x)
        out = self.decoder(latent_rep)
        return out, latent_rep


from typing import TypedDict


class Params(TypedDict):
    num_epochs: int
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: object
    loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    train_loader: torch.utils.data.DataLoader
    device: torch.device
    name: str


def train(params: Params):
    Path(f"./{params['name']}").mkdir(exist_ok=True)
    Path(f"./{params['name']}/images").mkdir(exist_ok=True)
    Path(f"./{params['name']}/checkpoints").mkdir(exist_ok=True)
    with open(f"{params['name']}/params.json", "w") as file:
        file.write(json.dumps(params, default=lambda o: str(o)))  # use `json.loads` to do the reverse

    for epoch in range(params["num_epochs"]):
        losses = []
        with progressbar.ProgressBar(max_value=len(params["train_loader"])) as bar:
            for batch_idx, data in enumerate(params["train_loader"]):
                inputs, targets = data

                params["optimizer"].zero_grad()

                outputs, latent = params["model"](inputs)
                loss = params["loss_func"](outputs, targets)
                loss.backward()
                params["optimizer"].step()
                losses.append(loss.item())
                if batch_idx == 0 and epoch % (params["num_epochs"] / 10) == 0:
                    torchvision.utils.save_image(outputs, f"{params['name']}/images/out_{epoch}.png")
                    torchvision.utils.save_image(latent[0, 0, :, :], f"{params['name']}/images/latent_{epoch}.png")

                bar.update(bar.value + 1)
        loss_metric = np.average(losses)
        print(f"Epoch: {epoch}, Loss: {loss_metric}")
        # TODO Validation
        scheduler.step(loss_metric)
        if epoch % int(params["num_epochs"] / 5) == 0:
            torch.save(params["model"].state_dict(), f"{params['name']}/checkpoints/checkpoint_{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Name of the model, used for storing files", required=True)
    parser.add_argument("-d", "--data", type=str, help="Path to folder containing data", required=True)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=20)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ImageReconstructionDataset(args.data, "png", device)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True)
    model = AutoEncoder(latent_dim=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    params: Params = {
        "num_epochs": args.epochs,
        "model": model.to(device),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_func": nn.MSELoss().to(device),
        "train_loader": train_loader,
        "device": device,
        "name": args.name
    }

    train(params)
