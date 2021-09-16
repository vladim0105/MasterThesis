import glob
import typing

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
        pil_img = Image.open(self.files[index]).convert('RGB')
        img = self.transform(pil_img).to(self.device)
        return img, img

    def __len__(self):
        return len(self.files)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.base = torchvision.models.resnet.resnet18(pretrained=True)
        self.base.fc = nn.Linear(in_features=512, out_features=latent_dim)

    def forward(self, x):
        return self.base(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=256)
        self.upconv1 = nn.ConvTranspose2d(in_channels=1, stride=(2, 2), out_channels=3, kernel_size=(2, 2))
        self.upconv2 = nn.ConvTranspose2d(in_channels=3, stride=(2, 2), out_channels=3, kernel_size=(2, 2))
        self.upconv3 = nn.ConvTranspose2d(in_channels=3, stride=(2, 2), out_channels=3, kernel_size=(2, 2))
        self.upconv4 = nn.ConvTranspose2d(in_channels=3, stride=(2, 2), out_channels=3, kernel_size=(2, 2))

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(-1, 1, 16, 16)
        x = self.upconv1(x)
        x = self.relu(x)
        x = self.upconv2(x)
        x = self.relu(x)
        x = self.upconv3(x)
        x = self.relu(x)
        x = self.upconv4(x)
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
    loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    train_loader: torch.utils.data.DataLoader
    device: torch.device
    name: str

def train(params: Params):
    for epoch in range(params["num_epochs"]):
        losses = []
        with progressbar.ProgressBar(max_value=len(params["train_loader"])) as bar:
            for batch_idx, data in enumerate(params["train_loader"]):
                inputs, targets = data

                params["optimizer"].zero_grad()

                outputs, _ = params["model"](inputs)
                loss = params["loss_func"](outputs, targets)
                loss.backward()
                params["optimizer"].step()
                losses.append(loss.item())
                if batch_idx == 0 and epoch % 5 == 0:
                    torchvision.utils.save_image(outputs, f"{params['name']}/images/out_{epoch}.png")
                    torch.save(params["model"].state_dict(), f"{params['name']}/checkpoints/checkpoint_{epoch}.pt")
                bar.update(bar.value + 1)
        print(f"Epoch: {epoch}, Loss: {np.average(losses)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Name of the model, used for storing files", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ImageReconstructionDataset(".", "png", device)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    model = AutoEncoder(latent_dim=50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    params: Params = {
        "num_epochs": 300,
        "model": model.to(device),
        "optimizer": optimizer,
        "loss_func": nn.MSELoss().to(device),
        "train_loader": train_loader,
        "device": device,
        "name": args.name
    }

    train(params)
