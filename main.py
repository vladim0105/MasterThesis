import torch
import torchvision.models as models
from torchsummary import summary
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import vis_utils

if __name__ == '__main__':
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    test = torch.Tensor(data)
    model = models.resnet18(pretrained=True)
    img = mpimg.imread("test_data/shopping_mall.jpg")
    vis_utils.save_feature_maps_of_image(img, model, vis_dim=4)
