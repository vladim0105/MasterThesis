import torch
import torchvision.models as models
from torchsummary import summary
from torchvision.transforms import transforms

import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import vis_utils
import model as m

if __name__ == '__main__':
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    test = torch.Tensor(data)
    model = models.resnet18(pretrained=True)
    data = {}
    model.avgpool.register_forward_hook(vis_utils.get_layer_activation(data))
    print(model)
    img = mpimg.imread("test_data/shopping_mall.jpg")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    model(img)
    # vis_utils.save_feature_maps_of_image(img, model, vis_dim=4)
    print(data["data"].squeeze().shape)
    test = m.CNNLayer()
    print(test(img).squeeze().shape)
