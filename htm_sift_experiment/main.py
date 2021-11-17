import os
from datetime import time, datetime

import mpld3
import numpy as np
import progressbar
from htm.bindings.sdr import SDR
from matplotlib import pyplot as plt

import model as m
import utils


def norm_anom(anom, frame):
    active_bits = (frame == 255).sum()
    ratio = active_bits / (frame.shape[0] * frame.shape[1])
    anom = anom * ratio
    return anom, ratio


def numpy_to_sdr(arr: np.ndarray):
    sdr = SDR(dimensions=arr.shape)
    sdr.dense = arr.tolist()
    return sdr

def sdr_to_numpy(sdr: SDR):
    return np.array(sdr.dense)


if __name__ == '__main__':
    import cv2.cv2 as cv2

    video_scale = 0.4
    sdr_vis_scale = 0.5
    vidcap = cv2.VideoCapture('../data/output_seg.mp4')
    success, frame = vidcap.read()
    scaled_frame_shape = (int(frame.shape[0] * video_scale), int(frame.shape[1] * video_scale))

    print(scaled_frame_shape)
    cv2.imwrite("frame.png", frame)
    sift = cv2.ORB_create(nfeatures=2000)

    sp_args = m.SpatialPoolerArgs()
    sp_args.seed = 2
    # sp_args.inputDimensions = (scaled_frame_shape[1] * scaled_frame_shape[0],)
    sp_args.inputDimensions = (scaled_frame_shape[1], scaled_frame_shape[0])
    sp_args.columnDimensions = (64, 32)
    sp_args.potentialPct = 1
    sp_args.potentialRadius = 3
    sp_args.localAreaDensity = 0.1
    sp_args.globalInhibition = False
    sp_args.wrapAround = False
    sp_args.synPermActiveInc = 0.001
    sp_args.synPermInactiveDec = 0.00001
    sp_args.stimulusThreshold = 3
    sp_args.boostStrength = 0
    sp_args.dutyCyclePeriod = 10000000

    tm_args = m.TemporalMemoryArgs()
    tm_args.columnDimensions = sp_args.columnDimensions
    tm_args.predictedSegmentDecrement = 0.000001
    tm_args.permanenceIncrement = 0.001
    tm_args.permanenceDecrement = 0.0000001
    tm_args.seed = sp_args.seed
    scaled_column_shape = (int(sp_args.columnDimensions[0]*sdr_vis_scale), int(sp_args.columnDimensions[1]*sdr_vis_scale))
    htm = m.HTMLayer(sp_args, tm_args)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (scaled_frame_shape[1], scaled_frame_shape[0]), 0)
    frame_id = 1
    anoms = []
    ratios = []
    diff = []
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = 50000
    with progressbar.ProgressBar(max_value=total) as bar:
        while success:
            frame = cv2.resize(frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            # frame_out = np.zeros(shape=(scaled_frame_shape[0], scaled_frame_shape[1]), dtype=np.uint8)
            # keypoints = sift.detect(frame, None)
            # for keypoint in keypoints:
            #     x, y = keypoint.pt
            #     frame_out[int(y)][int(x)] = 255
            frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Comment out to use SIFT
            frame_out = (frame_out > 200) * 255
            frame_out = frame_out.astype(np.uint8)
            sdr = numpy_to_sdr((frame_out == 255))
            pred, anom, sp_sdr = htm(sdr, learn=True)
            cv2.imwrite("test.png", sdr_to_numpy(sp_sdr)*255)
            _, ratio = norm_anom(anom, frame_out)
            anoms.append(anom)
            ratios.append(ratio*10)
            diff.append(anom-ratio)

            frame_sp_sdr = sdr_to_numpy(sp_sdr) * 255
            frame_sp_sdr = cv2.resize(frame_sp_sdr, dsize=(scaled_column_shape[0], scaled_column_shape[1]), interpolation=cv2.INTER_NEAREST)
            frame_out[0:scaled_column_shape[1], 0:scaled_column_shape[0]] = frame_sp_sdr
            frame_number = utils.text_phantom(str(bar.value), 12)
            frame_out[0:12, -(12 * 5):] = frame_number

            out.write(frame_out)
            success, frame = vidcap.read()
            bar.update(bar.value + 1)
            if bar.value == total:
                break

    fig = plt.figure(figsize=(40, 6))
    plt.plot(anoms, label="anom")
    plt.plot(ratios, label="ratio")
    plt.plot(diff, label="diff")
    plt.legend()
    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=25, fmt=".5f"))
    html_str = mpld3.fig_to_html(fig)
    html_file = open(f'fig_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html', "w")
    html_file.write(html_str)
    html_file.close()
