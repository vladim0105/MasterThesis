import glob
import json
import random
from copy import deepcopy

import numpy as np
from htm.bindings.encoders import RDSE_Parameters, RDSE
from matplotlib import pyplot as plt

import model as m
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
import torch.functional as F
import utils
import vis_utils
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == "__main__":
    model = torchvision.models.resnet.resnet34(pretrained=True).cuda()
    model.train(False)
    files = glob.glob(f"data/data/*.jpg")
    files.sort(key=natural_keys)
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    categories = utils.get_imagenet_classes("../../imagenet_classes.txt")
    activations = {}
    model.avgpool.register_forward_hook(vis_utils.get_layer_activation(activations, "avgpool"))

    encoder_params = RDSE_Parameters()
    encoder_params.seed = 11
    encoder_params.size = 500
    encoder_params.sparsity = 0.1
    encoder_params.resolution = 0.02
    encoder = RDSE(encoder_params)

    sp_args = m.SpatialPoolerArgs()
    sp_args.seed = encoder_params.seed
    sp_args.inputDimensions = (512*encoder_params.size,)
    sp_args.columnDimensions = (1000,)
    sp_args.potentialPct = 0.5
    sp_args.potentialRadius = 2048
    sp_args.localAreaDensity = 0.05
    sp_args.globalInhibition = True

    tm_args = m.TemporalMemoryArgs()
    tm_args.columnDimensions=sp_args.columnDimensions
    tm_args.predictedSegmentDecrement=0.001
    tm_args.permanenceIncrement=0.01
    tm_args.permanenceDecrement=0.001
    tm_args.seed = sp_args.seed

    htm_layer = m.HTMLayer(sp_args, tm_args)
    anomalies = []
    probs = []
    losses = []
    data = []
    scorp_idx = categories.index("scorpion")
    scorp_target = torch.zeros(size=(1, len(categories))).long().cuda()
    print(scorp_target.shape)
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    scorp_target[0][scorp_idx] = 1
    random.seed(5)
    #random.shuffle(files)
    counter = 0
    for file in files:
        name = file[10:]
        img = Image.open(file)
        input = transform(img).unsqueeze(0).cuda()
        torchvision.utils.save_image(input, f"data/data/input/input_{name}")
        output = model(input)
        probabilities = torch.functional.F.softmax(output[0], dim=0)
        loss = loss_func(output, torch.max(scorp_target, 1)[1]).item()
        top5_prob, top5_catid = torch.topk(probabilities, len(categories))
        prob_idx = (top5_catid == scorp_idx).nonzero().cpu().item()
        latent = activations["avgpool"].squeeze().cpu()
        if counter==500:
            latent = torch.zeros_like(latent)
        torchvision.utils.save_image(latent.reshape(16, 32), f"data/data/latent/latent_{name}")
        prob = top5_prob[prob_idx].cpu().item()
        probs.append(prob)
        losses.append(loss)
        htm_in = utils.float_array_to_sdr(latent, encoder)
        pred, anom, anomp = htm_layer(htm_in, True)

        anomalies.append(anom)

        dataline = f"{anom:.3f},{prob:.3f},{categories[top5_catid[0]]},{loss}"
        data.append(dataline)
        print(f"({counter}) {file} : {dataline}")
        counter += 1
    print(np.corrcoef(np.array(anomalies)[300:], np.array(probs)[300:]))
    print(np.corrcoef(np.array(anomalies)[300:], np.array(losses)[300:]))
    with open("probs.json", "w") as f:
        json.dump({"data": data}, f)
    xs = np.arange(1, len(anomalies) + 1, 1)
    inserted_anoms = [549+1, 500+1, 599+1]
    plt.plot(xs, anomalies, color="blue", label="HTM Anomaly Score")
    plt.vlines(inserted_anoms, 0, 1, color="red", label="Anomalies", alpha=0.5)
    plt.xlabel("Image #")
    plt.ylabel("Anomaly Score")
    plt.xlim(1, len(anomalies) + 5)
    plt.legend()

    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.show()
