import glob

import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
import torch.functional as F
import utils
import vis_utils
if __name__ == "__main__":
    model = torchvision.models.resnet.resnet18(pretrained=True).cuda()
    model.train(False)
    files = glob.glob(f"data/scorpion*.jpg")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    categories = utils.get_imagenet_classes("../../imagenet_classes.txt")
    activations = {}
    model.avgpool.register_forward_hook(vis_utils.get_layer_activation(activations))
    for file in files:
        img = Image.open(file)
        input = transform(img).unsqueeze(0).cuda()
        torchvision.utils.save_image(input, f"data/input_{file[5:]}")
        output = model(input)
        probabilities = torch.functional.F.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 3)
        torchvision.utils.save_image(activations["data"].reshape(16, 32), f"data/latent_{file[5:]}")
        #for i in range(top5_prob.size()[0]):
        #   print(categories[top5_catid[i]], top5_prob[i].cpu().item())