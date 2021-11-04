import argparse
import sys
from pathlib import Path

import numpy as np
import progressbar
import segmentation_models_pytorch as smp
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

import datasets
import utils


def test(params, args):
    with torch.no_grad():
        params["model"].load_state_dict(torch.load(f"./{args.name}/checkpoints/best_iou.pt"))
        params["model"].eval()
        pil_img = Image.open(args.test).convert('RGB')
        img, _ = params["val_loader"].dataset.transform(pil_img, None)
        img = img.to(params["device"])
        img = img.unsqueeze(0)  # Add batch dimension
        torchvision.utils.save_image(img[0], f"{params['name']}/aaa.png")
        out = params["model"](img)
        out=torch.where(out > 0.05, 1.0, 0.0)
        print(out.shape)
        print(f"Saving output to {params['name']}/result.png...")
        torchvision.utils.save_image(out[0].unsqueeze(1), f"{params['name']}/result.png", pad_value=1)
        print("Done")

def train(params, args):
    # Setup folders etc...
    Path(f"./{params['name']}").mkdir(exist_ok=True)
    Path(f"./{params['name']}/images").mkdir(exist_ok=True)
    Path(f"./{params['name']}/checkpoints").mkdir(exist_ok=True)
    Path(f"./{params['name']}/plots").mkdir(exist_ok=True)
    plot_data = {"train_loss": [], "val_loss": []}
    best_iou = 0
    print(f'Training on: {params["device"]}, using {args.workers} workers...')
    for epoch in range(params["num_epochs"]):
        train_losses = []
        params["model"].train()
        with progressbar.ProgressBar(max_value=len(params["train_loader"]), prefix="Training: ", fd=sys.stdout) as bar:
            for batch_idx, data in enumerate(params["train_loader"]):
                inputs, targets = data
                inputs = inputs.to(params["device"])
                targets = targets.to(params["device"])
                params["optimizer"].zero_grad()

                outputs = params["model"](inputs)
                loss = params["loss_func"](outputs, targets)
                loss.backward()
                params["optimizer"].step()
                train_losses.append(loss.item())
                if interactive:
                    bar.update(batch_idx)
                else:
                    if batch_idx % 1000 == 0:
                        bar.update(batch_idx)
        train_loss = np.average(train_losses)
        # Validation
        params["model"].eval()
        val_losses = []
        val_ious = []
        with progressbar.ProgressBar(max_value=len(params["val_loader"]), prefix="Validating: ", fd=sys.stdout) as bar:
            with torch.no_grad():
                for batch_idx, data in enumerate(params["val_loader"]):
                    inputs, targets = data
                    inputs = inputs.to(params["device"])
                    targets = targets.to(params["device"])

                    outputs = params["model"](inputs)
                    loss = params["loss_func"](outputs, targets)
                    segmentation = torch.where(outputs > 0.5, 1.0, 0.0)
                    iou = utils.iou_pytorch(segmentation.long(), targets.long())

                    val_losses.append(loss.item())
                    val_ious.append(iou.item())
                    # Save useful info during validation...
                    if batch_idx == 0 and epoch % 1 == 0:
                        out = segmentation[0]
                        out = out.unsqueeze(1)
                        torchvision.utils.save_image(out, f"{params['name']}/images/out_{epoch}.png", pad_value=1)
                        if epoch == 0:
                            torchvision.utils.save_image(inputs[0], f"{params['name']}/images/in.png")
                            torchvision.utils.save_image(targets[0].unsqueeze(1), f"{params['name']}/images/target.png")

                    if interactive:
                        bar.update(batch_idx)

        val_loss = np.average(val_losses)
        val_iou = np.average(val_ious)

        plot_data["val_loss"].append(val_loss)
        params["scheduler"].step(val_loss)
        if epoch % int(params["num_epochs"] / 5) == 0:
            torch.save(params["model"].state_dict(), f"{params['name']}/checkpoints/checkpoint_{epoch}.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            print("New best IoU, saving model...")
            torch.save(params["model"].state_dict(), f"{params['name']}/checkpoints/best_iou.pt")

        utils.log(f"Epoch: {epoch+1}, LR: {params['scheduler'].get_last_lr():.3f}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, IoU: {val_iou:.3f}",
                  f"{params['name']}/log.txt")
        print("-")


if __name__ == "__main__":
    interactive = sys.stdout.isatty()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Name of the model, used for storing files", required=True)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=20)
    parser.add_argument("-w", "--workers", type=int, help="Number of dataloader workers", default=1)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.01)
    parser.add_argument("--test", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    scat = ["person", "vehicle", "outdoor", "plant", "building", "sky", "ground", "solid", "floor"]
    model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet", classes=len(scat))
    model = torch.nn.DataParallel(model)
    dataloader_train = None
    if args.test is None:
        dataset_train = datasets.UAVIDSegmentationDataset("../../train2017",
                                                         "../../annotations/instances_train2017.json",
                                                         "../../annotations/stuff_train2017.json", scat)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, pin_memory=True,
                                      num_workers=args.workers, drop_last=True)

    dataset_val = datasets.COCOSegmentationDataset("../../val2017",
                                                   "../../annotations/instances_val2017.json",
                                                   "../../annotations/stuff_val2017.json", scat, val=True)

    dataloader_val = DataLoader(dataset_val, batch_size=args.batch, shuffle=False, pin_memory=True,
                                num_workers=args.workers, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {
        "num_epochs": args.epochs,
        "model": model.to(device),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_func": smp.losses.JaccardLoss(mode="multilabel").to(device),
        "train_loader": dataloader_train,
        "val_loader": dataloader_val,
        "device": device,
        "name": args.name
    }
    if args.test is not None:
        test(params, args)
    else:
        train(params, args)
