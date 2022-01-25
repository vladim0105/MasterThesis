import pickle
from datetime import time, datetime

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



def keypoints_to_bits(shape, kps):
    arr = np.zeros(shape=shape)
    for keypoint in kps:
        print(keypoint)
        x = int(keypoint[0][0])
        y = int(keypoint[0][1])
        arr[y, x] = 1
    return arr


def grid_tm(tms, sp_output: np.ndarray, grid_size, func, prev_sp_output: np.ndarray):
    anoms = np.zeros(shape=(len(tms), len(tms)))
    if prev_sp_output is None:
        prev_sp_output = np.ones_like(sp_output)
    colored_sdr_arr = np.zeros(shape=(sp_output.shape[0], sp_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(tms)):
        for j in range(len(tms[i])):
            tm = tms[i][j]
            val = sp_output[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size]
            sdr_cell = model.numpy_to_sdr(val)
            pred, anom, n_pred_cells = tm(sdr_cell, learn=True)
            prev_val = prev_sp_output[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size]
            if (prev_val == 0).all():
                anom = 0

            colored_sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size, 0] = int(
                60 * (1 - anom))
            colored_sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size, 1] = 255
            colored_sdr_arr[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size, 2] = 255 * (1 - val)
            anoms[i, j] = anom

    colored_sdr_arr = cv2.cvtColor(colored_sdr_arr, cv2.COLOR_HSV2BGR)
    return func(anoms.flatten()), colored_sdr_arr


def grid_sp(sps, sp_input: np.ndarray, in_grid_size, out_grid_size, empty_pattern, keypoint_detector):
    sp_output = np.zeros(shape=(out_grid_size * len(sps), out_grid_size * len(sps)))
    for i in range(len(sps)):
        for j in range(len(sps[i])):
            sp = sps[i][j]
            val = sp_input[i * in_grid_size: (i + 1) * in_grid_size, j * in_grid_size: (j + 1) * in_grid_size]
            # Check if empty
            if not (val == 1).any():
                val = empty_pattern
            sdr_cell = model.numpy_to_sdr(val)
            sp_cell_output = model.sdr_to_numpy(sp(sdr_cell, learn=True))
            sp_output[i * out_grid_size: (i + 1) * out_grid_size,
            j * out_grid_size: (j + 1) * out_grid_size] = sp_cell_output
    return sp_output


if __name__ == '__main__':
    import cv2.cv2 as cv2

    video_scale = 0.3
    sdr_vis_scale = 0.5
    vidcap = cv2.VideoCapture('../data/output_seg_2.mp4')
    vidcap = cv2.VideoCapture('../data/sperm_seg.mp4')
    success, frame = vidcap.read()
    frame = concat_seg(frame, success)
    scaled_frame_shape = (int(frame.shape[0] * video_scale), int(frame.shape[1] * video_scale))
    square_size = max(scaled_frame_shape[0], scaled_frame_shape[1])
    sp_grid_size = 16
    tm_grid_size = 8
    sparsity = 40  # How many ON bits per gridcell the encoding should produce
    empty_pattern = utils.random_bit_array(shape=(sp_grid_size, sp_grid_size), num_ones=sparsity)
    keypoint_detector = cv2.ORB_create(nfeatures=sparsity*5, edgeThreshold=0, fastThreshold=0, nlevels=20, patchSize=2)
    sp_args = m.SpatialPoolerArgs()
    sp_args.seed = 2
    # sp_args.inputDimensions = (scaled_frame_shape[1] * scaled_frame_shape[0],)
    sp_args.inputDimensions = (sp_grid_size, sp_grid_size)
    sp_args.columnDimensions = (tm_grid_size, tm_grid_size)
    sp_args.potentialPct = 0.05
    sp_args.potentialRadius = 5
    sp_args.localAreaDensity = 0.1
    sp_args.globalInhibition = False
    sp_args.wrapAround = False
    sp_args.synPermActiveInc = 0.01
    sp_args.synPermInactiveDec = 0.001
    sp_args.stimulusThreshold = 3
    sp_args.boostStrength = 0
    sp_args.dutyCyclePeriod = 5

    tm_args = m.TemporalMemoryArgs()

    tm_args.columnDimensions = (tm_grid_size, tm_grid_size)
    tm_args.predictedSegmentDecrement = 0.003
    tm_args.permanenceIncrement = 0.01
    tm_args.permanenceDecrement = 0.001
    tm_args.minThreshold = 1
    tm_args.activationThreshold = 2
    tm_args.cellsPerColumn = 32
    tm_args.seed = sp_args.seed
    scaled_column_shape = (
        int(256 * sdr_vis_scale), int(256 * sdr_vis_scale))
    sps = []
    tms = []
    # Spatial Pooler Init
    for i in range(square_size // sp_grid_size):
        sps_inner = []
        for j in range(square_size // sp_grid_size):
            sp_args.seed +=1
            sps_inner.append(model.SpatialPooler(sp_args))
        sps.append(sps_inner)
    # Temporal Memory Init
    for i in range(square_size // (2*tm_grid_size)):
        tms_inner = []
        for j in range(square_size // (2*tm_grid_size)):
            tms_inner.append(model.TemporalMemory(tm_args))
        tms.append(tms_inner)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (square_size, square_size), True)
    frame_id = 1
    anoms = []
    ratios = []
    diff = []
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #total = 500
    prev_sp_output = None
    with progressbar.ProgressBar(max_value=total, widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                                           progressbar.ETA()]) as bar:
        while success:
            frame = cv2.resize(frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            frame_out = frame
            frame_out, size = force_square(frame_out)
            frame_out = (frame_out > 200) * 255
            frame_out = frame_out.astype(np.uint8)
            sp_input = (frame_out == 255)[:, :, 0].astype(np.uint8)
            sp_output = grid_sp(sps, sp_input, sp_grid_size, tm_grid_size, empty_pattern, keypoint_detector)
            if bar.value == 1:
                cv2.imwrite("frame.png", frame)
                cv2.imwrite("sp_input.png", sp_input * 255)
                cv2.imwrite("sp_output.png", sp_output * 255)
            anom, colored_sp_output = grid_tm(tms, sp_output, tm_grid_size, np.mean, prev_sp_output)
            prev_sp_output = sp_output

            _, ratio = norm_anom(anom, frame_out)
            anoms.append(anom)
            ratios.append(ratio)
            diff.append(anom * ratio)

            colored_sp_output = cv2.resize(colored_sp_output, dsize=(scaled_column_shape[1], scaled_column_shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
            frame_out[0:scaled_column_shape[0], 0:scaled_column_shape[1], :] = colored_sp_output
            frame_number = utils.text_phantom(str(bar.value), 12)
            frame_out[0:12, -(12 * 5):] = frame_number
            out.write(frame_out)
            success, frame = vidcap.read()
            frame = concat_seg(frame, success)
            bar.update(bar.value + 1)
            if bar.value == total:
                break

    # fig, axs = plt.subplots(3, figsize=(6, 6))
    #
    # axs[0].plot(anoms, label="anom")
    # axs[0].set_title("Raw Anomaly Score")
    #
    # axs[1].plot(ratios, label="ratio")
    # axs[1].set_title("Ratio of active pixels")
    # axs[1].sharex(axs[0])
    # axs[2].plot(diff, label="diff")
    # axs[2].set_title("RAS*Ratio")
    # axs[2].sharex(axs[0])
    dump_data = {"anom_scores": anoms, "anom_markers": []}
    pickle.dump(dump_data, open(f'anoms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb'))
    pickle.dump(dump_data, open(f'anoms_latest.pkl', 'wb'))
