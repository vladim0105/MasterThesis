import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Adapted from https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
from torchvision.transforms import transforms


def get_conv_layers_and_weights(model: nn.Module) -> (list, list):
    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    return conv_layers, model_weights


def create_feature_maps(img: torch.Tensor, conv_layers: list) -> list:
    # pass the image through all the layers
    feature_maps = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        feature_maps.append(conv_layers[i](feature_maps[-1]))
    # make a copy of the `results`
    return feature_maps


def save_feature_maps(feature_maps: list, vis_dim=8):
    # visualize 64 features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(feature_maps)):
        fig = plt.figure(figsize=(30, 30))
        layer_viz = feature_maps[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        fig.suptitle(f"Layer {num_layer}", fontsize=100)
        for i, feature_map in enumerate(layer_viz):
            if i == vis_dim ** 2:  # we will visualize only vis_dim*vis_dim blocks from each layer
                break
            plt.subplot(vis_dim, vis_dim, i + 1)
            plt.imshow(feature_map, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"feature_maps/layer_{num_layer}.png")
        # plt.show()
        plt.close()


def save_feature_maps_of_image(img: np.ndarray, model, vis_dim=8):
    conv_layers, _ = get_conv_layers_and_weights(model)
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    feature_maps = create_feature_maps(img, conv_layers)
    save_feature_maps(feature_maps, vis_dim=vis_dim)
