import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import vis_utils

if __name__ == '__main__':
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    test = torch.Tensor(data)
    model = models.resnet18(pretrained=True)
    img = mpimg.imread("shopping_mall.jpg")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    convlayers, _ = vis_utils.get_conv_layers_and_weights(model)
    featuremaps = vis_utils.create_feature_maps(img, convlayers[:2])
    n = 50

    for i in range(63):
        featuremap = featuremaps[1][0][i].detach().numpy()
        #cutoff = -np.sort(-featuremap.flatten())[n]
        #featuremap = np.where(featuremap > cutoff, 255, 0)
        mpimg.imsave(f"feature_maps/{i}.png", featuremap)

    #vis_utils.save_feature_maps_of_image(img, model, vis_dim=4)