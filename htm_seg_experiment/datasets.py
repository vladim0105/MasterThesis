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
import utils


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

    def transform(self, _img, _mask, size=256):
        _img = torchvision.transforms.functional.equalize(_img)
        # Transform to tensor
        _img = torchvision.transforms.functional.to_tensor(_img)

        width, height = torchvision.transforms.functional.get_image_size(_img)

        if _mask is not None:
            _mask = torch.Tensor(_mask)

        if not self.val:
            # Random resize crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(_img, [0.5, 1], [1, 1])
            _img = torchvision.transforms.functional.resized_crop(_img, i, j, h, w, size=[size, size])
            _mask = torchvision.transforms.functional.resized_crop(_mask, i, j, h, w, size=[size, size])
            # Random horizontal flipping
            if random() > 0.5:
                _img = torchvision.transforms.functional.hflip(_img)
                _mask = torchvision.transforms.functional.hflip(_mask)
        else:
            # Make sure it is 256x256
            # Resize, keep aspect ratio
            resize = transforms.Resize(size=size)
            crop = transforms.CenterCrop(size=size)
            _img = resize(_img)
            _img = crop(_img)
            if _mask is not None:
                _mask = resize(_mask)
                _mask = crop(_mask)

        return _img, _mask

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
        anns_all = anns + anns_stuff

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


class UAVIDSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, num_crops=10, val=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.num_crops = num_crops
        self.val = val
        self.filenames_img = glob.glob(f"{image_dir}/**/Images/*.png", recursive=True)
        self.filenames_mask = glob.glob(f"{label_dir}/**/Labels/*.png", recursive=True)
        self.class_names = ["background", "building", "road", "tree", "low_vegetation", "car", "human"]
        self.rgb_to_class_idx = {(0, 0, 0): 0, (128, 0, 0): 1, (128, 64, 128): 2, (0, 128, 0): 3, (128, 128, 0): 4,
                                 (64, 0, 128): 5,
                                 (192, 0, 192): 5, (64, 64, 0): 6}

    def transform(self, pil_img, pil_mask, size=256):
        pil_img = torchvision.transforms.functional.equalize(pil_img)
        # Transform to tensor
        img = torchvision.transforms.functional.to_tensor(pil_img)
        normalize = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = normalize(img)
        mask_rgb = torchvision.transforms.functional.pil_to_tensor(pil_mask)
        mask = utils.label_rgb_mask(mask_rgb, self.rgb_to_class_idx).unsqueeze(0)
        crop_imgs = torch.zeros(size=(self.num_crops, 3, size, size))
        crop_masks = torch.zeros(size=(self.num_crops, 1, size, size)).long()
        for crop_id in range(self.num_crops):
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(256, 256))
            crop_img = torchvision.transforms.functional.resized_crop(img, i, j, h, w, size=[size, size])
            crop_mask = torchvision.transforms.functional.resized_crop(mask, i, j, h, w, size=[size, size])

            if random() > 0.5:
                crop_img = torchvision.transforms.functional.hflip(crop_img)
                crop_mask = torchvision.transforms.functional.hflip(crop_mask)

            crop_imgs[crop_id] = crop_img
            crop_masks[crop_id] = crop_mask

        return crop_imgs, crop_masks

    def val_transform(self, pil_img, pil_mask, deterministic=False, size=256):
        if not deterministic:
            return self.transform(pil_img, pil_mask, size)
        crop = torchvision.transforms.CenterCrop(size=size)

        pil_img = torchvision.transforms.functional.equalize(pil_img)
        # Transform to tensor
        img = torchvision.transforms.functional.to_tensor(pil_img)
        normalize = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = normalize(img)
        img = crop(img).unsqueeze(0)
        mask = None
        if pil_mask is not None:
            mask_rgb = torchvision.transforms.functional.pil_to_tensor(pil_mask)
            mask = utils.label_rgb_mask(mask_rgb, self.rgb_to_class_idx).unsqueeze(0)
            mask = crop(mask).unsqueeze(0)

        return img, mask

    def test_transform(self, pil_img, size=256):
        resize = torchvision.transforms.Resize(size=size)
        crop = torchvision.transforms.CenterCrop(size=size)
        normalize = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        pil_img = torchvision.transforms.functional.equalize(pil_img)
        img = torchvision.transforms.functional.to_tensor(pil_img)
        img = normalize(img)
        img = resize(img)
        img = crop(img)

        return img

    def __getitem__(self, index):
        pil_img = Image.open(self.filenames_img[index]).convert("RGB")
        pil_mask = Image.open(self.filenames_mask[index]).convert("RGB")
        if self.val:
            return self.val_transform(pil_img, pil_mask, index == 0)

        # crop_imgs, crop_masks = self.transform(pil_img, pil_mask)

        crop_imgs, crop_masks = self.transform(pil_img, pil_mask)
        return crop_imgs, crop_masks

    def __len__(self):
        return len(self.filenames_img)


if __name__ == "__main__":
    # scat = ["person", "vehicle", "outdoor", "plant", "building", "sky", "ground", "solid", "floor"]
    # img, mask = COCOSegmentationDataset("../../val2017", "../../annotations/instances_val2017.json",
    #                                     "../../annotations/stuff_val2017.json", scat).__getitem__(10)
    # torchvision.utils.save_image(img, "img.jpg")
    # mask = mask.unsqueeze(1)
    # torchvision.utils.save_image(mask, "mask.jpg", pad_value=1)
    ds = UAVIDSegmentationDataset("../../uavid/uavid_train", "../../uavid/uavid_train")

    crop_imgs, crop_masks = ds.__getitem__(10)
    print(crop_imgs.shape)
    print(crop_imgs.dtype)
    plt.imsave("test.png", np.array(crop_masks[0].permute(1, 2,0).squeeze()))
    torchvision.utils.save_image(crop_imgs, "test_in.png")
    #torchvision.utils.save_image(crop_masks, "test.png")
