
import progressbar
import dataset_to_video
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import vis_utils
import model as m
from FrameDataset import FrameDataset
import numpy as np
from datetime import datetime
if __name__ == '__main__':
    np.random.seed(2)
    dataset = FrameDataset("video_1.npy")
    anomalies = []
    anomalies_likelihood=[]
    preds = []

    sp_args = m.SpatialPoolerArgs()
    sp_args.seed = 2
    sp_args.inputDimensions = (140*120, )
    sp_args.columnDimensions = (1000,)
    sp_args.potentialPct = 0.5
    sp_args.potentialRadius = 2048
    sp_args.localAreaDensity = 0.05
    sp_args.globalInhibition = True

    tm_args = m.TemporalMemoryArgs()
    tm_args.columnDimensions = sp_args.columnDimensions
    tm_args.predictedSegmentDecrement = 0.001
    tm_args.permanenceIncrement = 0.1
    tm_args.permanenceDecrement = 0.001
    tm_args.seed = sp_args.seed
    print("aaa")
    htm_layer = m.HTMLayer(sp_args, tm_args)
    dataset_to_video.dataset_to_video(dataset.dataset, "test.avi")
    # First, expose the HTM network to the video several times
    n=10
    with progressbar.ProgressBar(max_value=n * len(dataset)) as bar:
        for j in range(n):
            for i in range(len(dataset)):
                sample = dataset[i].flatten()
                sdr = utils.tensor_to_sdr(sample)
                pred, anom, _ = htm_layer(sdr, True)
                anomalies.append(anom)
                bar.update(bar.value+1)
    dataset_to_video.dataset_to_video(dataset.dataset, "test.avi")
    #Then expose it to the anomalies
    #dataset = FrameDataset("video_2.npy")
    dataset.create_anomalies(5, 20, True)
    dataset_to_video.dataset_to_video(dataset.dataset, "test2.avi")
    for i in range(len(dataset)):
        sample = dataset[i].flatten()
        sdr = utils.tensor_to_sdr(sample)
        pred, anom, _ = htm_layer(sdr, True)
        preds.append(pred)
        anomalies.append(anom)
    print("AAAA")
    #plt.plot(dataset.anomalies, color="red", label="anomalies")
    plt.plot(anomalies, color="blue", label="htm anomalies")
    plt.ylim([0, 1.2])
    plt.xlabel("Frame #")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.savefig(f"plots/{datetime.now()}.png")
    plt.show()
