import os
import pickle
from datetime import time, datetime

import mpld3
import numpy as np
import progressbar
from htm.bindings.sdr import SDR
from matplotlib import pyplot as plt

import model
import model as m
import utils


def concat_seg(frame, success):
    if not success:
        return None
    seg_1 = frame[:frame.shape[0] // 2, :]
    seg_2 = frame[frame.shape[0] // 2:, :]
    # out = np.maximum(seg_1, seg_2)
    out = seg_1
    return out


def force_square(frame):
    width = frame.shape[0]
    height = frame.shape[1]
    val = max(width, height)
    out = np.zeros(shape=(val, val, 3))
    out[:width, :height, :] = frame
    return out, val


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


def grid_tm(tms, sp_output, grid_size, func, prev_sp_output):
    anoms = np.zeros(shape=(len(tms), len(tms)))
    sdr_arr = sdr_to_numpy(sp_output)
    prev_sdr_arr = np.ones_like(sdr_arr)
    if prev_sp_output is not None:
        prev_sdr_arr = sdr_to_numpy(prev_sp_output)
    colored_sdr_arr = np.zeros(shape=(sdr_arr.shape[0], sdr_arr.shape[1], 3), dtype=np.uint8)
    for i in range(len(tms)):
        for j in range(len(tms[i])):
            tm = tms[i][j]
            val = sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size]
            sdr_cell = numpy_to_sdr(val)
            pred, anom = tm(sdr_cell, learn=True)
            prev_val = prev_sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size]
            if (prev_val == 0).all():
                anom = 0

            colored_sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size, 0] = int(
                60 * (1 - anom))
            colored_sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size, 1] = 255
            colored_sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size, 2] = 255 * (1 - val)
            anoms[i, j] = anom

    colored_sdr_arr = cv2.cvtColor(colored_sdr_arr, cv2.COLOR_HSV2BGR)
    return func(anoms.flatten()), colored_sdr_arr


if __name__ == '__main__':
    import cv2.cv2 as cv2

    video_scale = 0.3
    sdr_vis_scale = 0.5
    vidcap = cv2.VideoCapture('../data/output_seg_2.mp4')
    success, frame = vidcap.read()
    frame = concat_seg(frame, success)
    scaled_frame_shape = (int(frame.shape[0] * video_scale), int(frame.shape[1] * video_scale))

    print(scaled_frame_shape)
    cv2.imwrite("frame.png", frame)
    sift = cv2.ORB_create(nfeatures=2000)

    sp_args = m.SpatialPoolerArgs()
    sp_args.seed = 2
    # sp_args.inputDimensions = (scaled_frame_shape[1] * scaled_frame_shape[0],)
    sp_args.inputDimensions = (384, 384)
    sp_args.columnDimensions = (192, 192)
    sp_args.potentialPct = 0.05
    sp_args.potentialRadius = 5
    sp_args.localAreaDensity = 0.1
    sp_args.globalInhibition = False
    sp_args.wrapAround = False
    sp_args.synPermActiveInc = 0.01
    sp_args.synPermInactiveDec = 0.000001
    sp_args.stimulusThreshold = 3
    sp_args.boostStrength = 0
    sp_args.dutyCyclePeriod = 10000000

    grid_size = 16
    grid = True
    tm_args = m.TemporalMemoryArgs()
    if grid:
        tm_args.columnDimensions = (grid_size, grid_size)
    else:
        tm_args.columnDimensions = sp_args.columnDimensions
    tm_args.predictedSegmentDecrement = 0.000001
    tm_args.permanenceIncrement = 0.001
    tm_args.permanenceDecrement = 0.000001
    tm_args.minThreshold = 1
    tm_args.activationThreshold = 1
    tm_args.seed = sp_args.seed
    scaled_column_shape = (
        int(sp_args.columnDimensions[0] * sdr_vis_scale), int(sp_args.columnDimensions[1] * sdr_vis_scale))
    sp = model.SpatialPooler(sp_args)
    tm = []
    if grid:
        for i in range(sp_args.columnDimensions[0] // grid_size):
            tms_inner = []
            for j in range(sp_args.columnDimensions[1] // grid_size):
                tms_inner.append(model.TemporalMemory(tm_args))
            tm.append(tms_inner)
    else:
        tm = model.TemporalMemory(tm_args)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (384, 384), True)
    frame_id = 1
    anoms = []
    ratios = []
    diff = []
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #total = 500
    prev_frame_sp_sdr = None
    with progressbar.ProgressBar(max_value=total) as bar:
        while success:
            frame = cv2.resize(frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            # frame_out = np.zeros(shape=(scaled_frame_shape[0], scaled_frame_shape[1]), dtype=np.uint8)
            # keypoints = sift.detect(frame, None)
            # for keypoint in keypoints:
            #     x, y = keypoint.pt
            #     frame_out[int(y)][int(x)] = 255
            frame_out = frame
            frame_out, size = force_square(frame_out)
            frame_out = (frame_out > 200) * 255
            frame_out = frame_out.astype(np.uint8)
            sdr = numpy_to_sdr((frame_out == 255)[:, :, 0])
            sp_sdr = sp(sdr, learn=True)
            frame_sp_sdr = sdr_to_numpy(sp_sdr)

            if grid:
                anom, colored_sp_sdr = grid_tm(tm, sp_sdr, grid_size, np.mean, prev_frame_sp_sdr)
                prev_frame_sp_sdr = sp_sdr
                frame_sp_sdr = colored_sp_sdr
            else:
                _, anom = tm(sp_sdr, learn=True)
                frame_sp_sdr = np.repeat(frame_sp_sdr[..., None], 3, axis=2)
            _, ratio = norm_anom(anom, frame_out)
            anoms.append(anom)
            ratios.append(ratio)
            diff.append(anom * ratio)

            # frame_sp_sdr = (frame_sp_sdr - 1) * (-1)  # Invert it to make it easier to see
            # frame_sp_sdr = frame_sp_sdr * 255
            frame_sp_sdr = cv2.resize(frame_sp_sdr, dsize=(scaled_column_shape[1], scaled_column_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            frame_out[0:scaled_column_shape[0], 0:scaled_column_shape[1], :] = frame_sp_sdr
            if bar.value == 0:
                cv2.imwrite("test.png", frame_out[0:scaled_column_shape[0], 0:scaled_column_shape[1], :])
            frame_number = utils.text_phantom(str(bar.value), 12)
            frame_out[0:12, -(12 * 5):] = frame_number
            out.write(frame_out)
            success, frame = vidcap.read()
            frame = concat_seg(frame, success)
            bar.update(bar.value + 1)
            if bar.value == total:
                break

    fig, axs = plt.subplots(3, figsize=(6, 6))

    axs[0].plot(anoms, label="anom")
    axs[0].set_title("Raw Anomaly Score")

    axs[1].plot(ratios, label="ratio")
    axs[1].set_title("Ratio of active pixels")
    axs[1].sharex(axs[0])
    axs[2].plot(diff, label="diff")
    axs[2].set_title("RAS*Ratio")
    axs[2].sharex(axs[0])
    pickle.dump(anoms, open(f'anoms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb'))
