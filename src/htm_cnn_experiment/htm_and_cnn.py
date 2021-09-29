from torchvision.transforms import transforms
import progressbar
import dataset_to_video
import utils
import matplotlib.pyplot as plt
import model as m
from FrameDataset import FrameDataset
from htm.encoders.rdse import RDSE, RDSE_Parameters
import numpy as np
from datetime import datetime
if __name__ == '__main__':
    seed = 1
    np.random.seed(seed+1)
    dataset = FrameDataset("video_1.npy")

    anomalies = []
    anomalies_likelihood=[]
    preds = []
    cnn_layer = m.CNNLayer()
    htm_layer = m.HTMLayer(shape=[512000], columnDimensions=[1000], seed=seed)
    dataset_to_video.dataset_to_video(dataset.dataset, "test.avi")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
    ])

    encoder_params = RDSE_Parameters()
    encoder_params.seed = seed
    encoder_params.size = 1000
    encoder_params.sparsity = 0.1
    encoder_params.resolution = 0.001
    encoder = RDSE(encoder_params)
    # First, expose the HTM network to the video several times
    n=2
    print("Training...")
    with progressbar.ProgressBar(max_value=n * len(dataset)) as bar:
        for j in range(n):
            for i in range(len(dataset)):
                sample = dataset[i]
                sample = transform(sample)
                sample = sample.unsqueeze(0)  # Add batch dimension
                out = cnn_layer(sample.cuda()).squeeze().detach().cpu().numpy()
                out = utils.float_array_to_sdr(out, encoder)
                htm_layer(out, True)
                bar.update(bar.value+1)
    dataset_to_video.dataset_to_video(dataset.dataset, "test.avi")
    #Then expose it to the anomalies
    #dataset = FrameDataset("video_2.npy")
    dataset.create_anomalies(5, 20, True)
    dataset_to_video.dataset_to_video(dataset.dataset, "test2.avi")
    print("Testing...")
    with progressbar.ProgressBar(max_value=len(dataset)) as bar:
        for i in range(len(dataset)):
            sample = dataset[i]
            sample = transform(sample)
            sample = sample.unsqueeze(0)  # Add batch dimension
            out = cnn_layer(sample.cuda()).squeeze().detach().cpu().numpy()
            out = utils.float_array_to_sdr(out, encoder)
            pred, anom, anomp = htm_layer(out, False)
            preds.append(pred)
            anomalies.append(anom)
            anomalies_likelihood.append(anomp)
            bar.update(bar.value+1)
    plt.plot(dataset.anomalies, color="red", label="anomalies")
    plt.plot(anomalies, color="blue", label="htm anomalies")
    #plt.plot(anomalies_likelihood, color="green", label="htm likelihood anomalies")
    plt.ylim([0, 1.2])
    plt.xlabel("Frame #")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.savefig(f"plots/{datetime.now()}.png")
    plt.show()
