from random import random

import matplotlib.pyplot as plt
import torch
import torchvision.utils
from torch.utils.data.dataset import T_co
import torchvision.transforms.functional
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import json
from pycocotools.coco import COCO
import numpy as np


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


class COCOSegmentationDataset(Dataset):

    def __init__(self, imageDir, annotationDir, annotationStuffDir, supercats, val=False):
        self.imageDir = imageDir
        print("Coco:")
        self.coco = COCO(annotation_file=annotationDir)
        print("Coco stuff:")
        self.coco_stuff = COCO(annotation_file=annotationStuffDir)
        self.cats = self.coco.loadCats(self.coco.getCatIds()) + self.coco_stuff.loadCats(self.coco_stuff.getCatIds())
        self.supercats = supercats
        self.imgIds = list(self.coco.imgs.keys())
        self.val = val

        self.catIdtoSupercat = {}
        for c in self.cats:
            try:
                self.catIdtoSupercat[c["id"]] = self.supercats.index(c["supercategory"])
            except:
                self.catIdtoSupercat[c["id"]] = -1

    def transform(self, image, mask):
        # Transform to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        mask = torch.Tensor(mask)
        # Resize, keep aspect ratio
        resize = transforms.Resize(size=256 if self.val else 512)
        image = resize(image)
        mask = resize(mask)
        if not self.val:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(256, 256))
            image = torchvision.transforms.functional.crop(image, i, j, h, w)
            mask = torchvision.transforms.functional.crop(mask, i, j, h, w)
            # Random horizontal flipping
            if random() > 0.5:
                image = torchvision.transforms.functional.hflip(image)
                mask = torchvision.transforms.functional.hflip(mask)
        else:
            # Make sure it is 256x256
            crop = transforms.CenterCrop(size=256)
            image = crop(image)
            mask = crop(mask)

        return image, mask

    def getClassName(self, classID):
        for i in range(len(self.cats)):
            if self.cats[i]['id'] == classID:
                return self.cats[i]['name']
        return "None"

    def __getitem__(self, index):
        id = self.imgIds[index]
        file_name = self.coco.imgs[id]["file_name"]
        img_path = f"{self.imageDir}/{file_name}"
        pil_img = Image.open(img_path).convert('RGB')
        anns = self.coco.imgToAnns[id]
        anns_stuff = self.coco_stuff.imgToAnns[id]
        anns_all = anns+anns_stuff

        mask = np.zeros((len(self.supercats), pil_img.size[1], pil_img.size[0]))
        for i in range(len(anns_all)):
            ann = anns_all[i]
            scat_id = self.catIdtoSupercat[ann["category_id"]]
            if scat_id == -1:
                continue
            mask[scat_id] = np.maximum(self.coco.annToMask(ann), mask[scat_id])
        img, mask = self.transform(pil_img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgIds)


if __name__ == "__main__":
    scat = ["person", "vehicle", "outdoor", "plant", "building", "sky", "ground", "solid", "floor"]
    img, mask = COCOSegmentationDataset("../../val2017", "../../annotations/instances_val2017.json",
                                        "../../annotations/stuff_val2017.json", scat).__getitem__(10)
    torchvision.utils.save_image(img, "img.jpg")
    mask = mask.unsqueeze(1)
    torchvision.utils.save_image(mask, "mask.jpg", pad_value=1)
