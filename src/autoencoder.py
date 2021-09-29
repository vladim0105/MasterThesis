import glob
import json
import random
import typing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import torch
import torchvision.models.resnet
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import T_co
from PIL import Image
from torchvision.transforms import transforms

from src.htm_cnn_experiment import utils
from piqa import SSIM


class ResidualWrapper(nn.Module):
    def __init__(self, module: nn.Module, res_op=None):
        super().__init__()
        self.module = module
        self.res_op = res_op

    def forward(self, inputs):
        res_out = inputs
        if self.res_op is not None:
            res_out = self.res_op(res_out)
        return self.module(inputs) + res_out


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


class ImageReconstructionDataset(Dataset):
    def __init__(self, path, extension, device):
        self.files = glob.glob(f"{path}/*.{extension}")
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])
        self.device = device
        print(f"Dataset found {len(self.files)} images!")

    def __getitem__(self, index) -> T_co:
        pil_img = Image.open(self.files[index])
        img = self.transform(pil_img)
        return img, img.clone()

    def __len__(self):
        return len(self.files)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.block1 = self.create_block(1, 2)
        self.block2 = self.create_block(2, 3)
        self.block3 = self.create_block(3, 4, dp=0)
        self.pool = nn.MaxPool2d(2, 2)

    def create_block(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, dp=0, residual=True):
        sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, stride=(2, 2), out_channels=out_channels, kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=dp)
        )
        if residual:
            block = ResidualWrapper(
                sequence,
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(2, 2))
                # 1x1 conv to adjust the residual channels
            )
        else:
            block = sequence
        return block

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.block1 = self.create_block(4, 3)
        self.block2 = self.create_block(3, 2)
        self.block3 = self.create_block(2, 1, activation=nn.Sigmoid, dp=0)
        self.sigmoid = nn.Sigmoid()

    def create_block(self, in_channels, out_channels, activation: typing.Type[nn.Module] = nn.LeakyReLU, dp=0.05):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, stride=(2, 2), out_channels=out_channels, kernel_size=(2,2)),
            nn.BatchNorm2d(num_features=out_channels),
            activation(),
            nn.Dropout(p=dp)
        )

        return block

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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
    val_loader: torch.utils.data.DataLoader
    device: torch.device
    name: str


def train(params: Params):
    plot_data = {"train_loss": [], "val_loss": []}
    for epoch in range(params["num_epochs"]):
        train_losses = []
        params["model"].train()
        with progressbar.ProgressBar(max_value=len(params["train_loader"]), prefix="Training: ") as bar:
            for batch_idx, data in enumerate(params["train_loader"]):
                inputs, targets = data
                inputs = inputs.to(params["device"])
                targets = targets.to(params["device"])
                params["optimizer"].zero_grad()

                outputs, latent = params["model"](inputs)
                loss = params["loss_func"](outputs, targets)
                loss.backward()
                params["optimizer"].step()
                train_losses.append(loss.item())
                bar.update(bar.value + 1)
        train_loss = np.average(train_losses)
        plot_data["train_loss"].append(train_loss)
        # Validation
        params["model"].eval()
        val_losses = []
        with progressbar.ProgressBar(max_value=len(params["val_loader"]), prefix="Validating: ") as bar:
            for batch_idx, data in enumerate(params["val_loader"]):
                inputs, targets = data
                inputs = inputs.to(params["device"])
                targets = targets.to(params["device"])

                outputs, latent = params["model"](inputs)
                loss = params["loss_func"](outputs, targets)
                val_losses.append(loss.item())
                # Save useful info during validation...
                if batch_idx == 0 and epoch % (params["num_epochs"] / 10) == 0:
                    torchvision.utils.save_image(outputs[0, :, :, :], f"{params['name']}/images/out_{epoch}.png")
                    latent_image = latent.transpose(1, 0)
                    torchvision.utils.save_image(latent_image,
                                                 f"{params['name']}/images/latent_{epoch}.png",
                                                 normalize=True,
                                                 nrow=int(np.sqrt(latent_image.shape[0])),
                                                 padding=0
                                                 )
                    if epoch == 0:
                        torchvision.utils.save_image(targets[0, :, :, :], f"{params['name']}/images/in.png")

                bar.update(bar.value + 1)

        val_loss = np.average(val_losses)
        plot_data["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        if epoch % int(params["num_epochs"] / 5) == 0:
            torch.save(params["model"].state_dict(), f"{params['name']}/checkpoints/checkpoint_{epoch}.pt")
        utils.log(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}", f"{params['name']}/log.txt")
        print("-")

    # Plot loss graph
    plt.yscale("log")
    plt.plot(plot_data["train_loss"], label="Training")
    plt.plot(plot_data["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Loss")
    plt.savefig(f"{params['name']}/plots/loss.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Name of the model, used for storing files", required=True)
    parser.add_argument("-d", "--data", type=str, help="Path to folder containing data", required=True)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=20)
    parser.add_argument("-w", "--workers", type=int, help="Number of dataloader workers", default=1)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.01)
    parser.add_argument("--extension", type=str, help="File extension for data", default="jpg")
    parser.add_argument("--ssim", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = ImageReconstructionDataset(args.data, args.extension, device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset=dataset, lengths=[train_size, val_size])

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = AutoEncoder(latent_dim=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=4)

    params: Params = {
        "num_epochs": args.epochs,
        "model": model.to(device),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_func": SSIMLoss(n_channels=1).to(device) if args.ssim else nn.MSELoss().to(device),
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": device,
        "name": args.name
    }
    # Setup folders etc...
    Path(f"./{params['name']}").mkdir(exist_ok=True)
    Path(f"./{params['name']}/images").mkdir(exist_ok=True)
    Path(f"./{params['name']}/checkpoints").mkdir(exist_ok=True)
    Path(f"./{params['name']}/plots").mkdir(exist_ok=True)
    with open(f"{params['name']}/params.json", "w") as file:
        file.write(json.dumps(params, default=lambda o: str(o)))  # use `json.loads` to do the reverse

    train(params)
