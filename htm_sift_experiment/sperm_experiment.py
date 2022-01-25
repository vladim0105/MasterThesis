import pickle
from datetime import time, datetime

import numpy as np
import progressbar
from htm.bindings.sdr import SDR
from matplotlib import pyplot as plt

import model
import model as m
import utils
import cv2.cv2 as cv2


def concat_seg(frame, success):
    if not success:
        return None
    seg_1 = frame[:frame.shape[0] // 2, :]
    seg_2 = frame[frame.shape[0] // 2:, :]
    # out = np.maximum(seg_1, seg_2)
    out = seg_1
    return out


def get_divisible_shape(current_shape, cell_size):
    width = current_shape[0]
    height = current_shape[1]
    new_width = (width + cell_size) - (width % cell_size)
    new_height = (height + cell_size) - (height % cell_size)
    return new_width, new_height


def force_divisible(frame, cell_size):
    new_width, new_height = get_divisible_shape(frame.shape, cell_size)
    out = np.zeros(shape=(new_width, new_height, 3))
    out[:frame.shape[0], :frame.shape[1], :] = frame
    return out


def keypoints_to_bits(shape, kps):
    arr = np.zeros(shape=shape)
    for keypoint in kps:
        print(keypoint)
        x = int(keypoint[0][0])
        y = int(keypoint[0][1])
        arr[y, x] = 1
    return arr


if __name__ == '__main__':
    pass

    video_scale = 0.3
    sdr_vis_scale = 1
    vidcap = cv2.VideoCapture('../data/sperm_seg3.mp4')
    success, frame = vidcap.read()
    frame = concat_seg(frame, success)
    scaled_frame_shape = (int(frame.shape[0] * video_scale), int(frame.shape[1] * video_scale))

    sp_grid_size = 16
    tm_grid_size = 8
    new_width, new_height = get_divisible_shape(scaled_frame_shape, sp_grid_size)

    sp_args = m.SpatialPoolerArgs()
    sp_args.seed = 2
    sp_args.inputDimensions = (sp_grid_size, sp_grid_size)
    sp_args.columnDimensions = (tm_grid_size, tm_grid_size)
    sp_args.potentialPct = 0.2
    sp_args.potentialRadius = 5
    sp_args.localAreaDensity = 0.1
    sp_args.globalInhibition = False
    sp_args.wrapAround = False
    sp_args.synPermActiveInc = 0.01
    sp_args.synPermInactiveDec = 0.001
    sp_args.stimulusThreshold = 1
    sp_args.boostStrength = 0
    sp_args.dutyCyclePeriod = 2500

    tm_args = m.TemporalMemoryArgs()

    tm_args.columnDimensions = (tm_grid_size, tm_grid_size)
    tm_args.predictedSegmentDecrement = 0.0005
    tm_args.permanenceIncrement = 0.01
    tm_args.permanenceDecrement = 0.001
    tm_args.minThreshold = 1
    tm_args.activationThreshold = 2
    tm_args.cellsPerColumn = 4
    tm_args.seed = sp_args.seed

    grid_htm = model.GridHTM((new_width, new_height), sp_grid_size, tm_grid_size, sp_args, tm_args, sparsity=5, aggr_func=np.mean)

    scaled_sdr_shape = (
        int(new_width * sdr_vis_scale), int(new_height * sdr_vis_scale))

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (new_height, new_width*2), True)
    anoms = []
    l1_scores = []
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #total = 250
    prev_encoded_input = None
    with progressbar.ProgressBar(max_value=total, widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                                           progressbar.ETA()]) as bar:
        while success:
            # Encode
            frame = cv2.resize(frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            frame = frame
            frame = force_divisible(frame, sp_grid_size)
            frame = (frame > 200) * 255
            frame = frame.astype(np.uint8)
            encoded_input = (frame == 255)[:, :, 0].astype(np.uint8)
            # L1 Score
            if prev_encoded_input is not None:
                l1_score = np.abs(encoded_input.astype(int) - prev_encoded_input.astype(int)).sum()
                l1_scores.append(l1_score)
            prev_encoded_input = encoded_input
            # Run HTM
            anom, colored_sp_output = grid_htm(encoded_input)
            anoms.append(anom)
            # Create output
            frame_out = np.zeros(shape=(frame.shape[0]*2, frame.shape[1], 3), dtype=np.uint8)
            colored_sp_output = cv2.resize(colored_sp_output, dsize=(scaled_sdr_shape[1], scaled_sdr_shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            frame_out[frame.shape[0]:frame.shape[0]+scaled_sdr_shape[0], 0:, :] = frame
            frame_out[0: frame.shape[0], 0:] = colored_sp_output
            frame_number = utils.text_phantom(str(bar.value), 12)
            frame_out[0:12, -(12 * 5):] = frame_number
            out.write(frame_out)

            # Get next frame
            prev_frame = frame
            success, frame = vidcap.read()
            frame = concat_seg(frame, success)

            bar.update(bar.value + 1)
            if bar.value == total:
                break

    dump_data = {"anom_scores": anoms, "anom_markers": [1510, 7540, 9000], "l1_scores": l1_scores}
    pickle.dump(dump_data, open(f'anoms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb'))
    pickle.dump(dump_data, open(f'anoms_latest.pkl', 'wb'))
