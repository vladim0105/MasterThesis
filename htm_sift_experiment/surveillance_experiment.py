import argparse
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("name", type=str, required=True)
    args = argparser.parse_args()


    video_scale = 0.3
    sdr_vis_scale = 1
    vidcap = cv2.VideoCapture('../data/output_seg_2.mp4')
    success, orig_frame = vidcap.read()
    orig_frame = concat_seg(orig_frame, success)
    scaled_frame_shape = (int(orig_frame.shape[0] * video_scale), int(orig_frame.shape[1] * video_scale))

    sp_grid_size = 32
    tm_grid_size = 16
    new_width, new_height = get_divisible_shape(scaled_frame_shape, sp_grid_size)

    sp_args = m.SpatialPoolerArgs()
    sp_args.seed = 2
    sp_args.inputDimensions = (sp_grid_size, sp_grid_size)
    sp_args.columnDimensions = (tm_grid_size, tm_grid_size)
    sp_args.potentialPct = 0.2
    sp_args.potentialRadius = 5
    sp_args.localAreaDensity = 0.05
    sp_args.globalInhibition = True
    sp_args.wrapAround = False
    sp_args.synPermActiveInc = 0.01
    sp_args.synPermInactiveDec = 0.00001
    sp_args.stimulusThreshold = 3
    sp_args.boostStrength = 0
    sp_args.dutyCyclePeriod = 2500

    tm_args = m.TemporalMemoryArgs()

    tm_args.columnDimensions = (tm_grid_size, tm_grid_size)
    tm_args.predictedSegmentDecrement = 0.001
    tm_args.permanenceIncrement = 0.01
    tm_args.permanenceDecrement = 0.001
    tm_args.minThreshold = 10
    tm_args.activationThreshold = 10
    tm_args.cellsPerColumn = 32
    tm_args.seed = sp_args.seed

    grid_htm = model.GridHTM((new_width, new_height), sp_grid_size, tm_grid_size, sp_args, tm_args, min_sparsity=10, sparsity=15, aggr_func=np.mean, temporal_size=15)

    scaled_sdr_shape = (
        int(new_width * sdr_vis_scale), int(new_height * sdr_vis_scale))

    out = cv2.VideoWriter(f'surveillance_results/{args.name}', cv2.VideoWriter_fourcc(*'mp4v'), 10,
                          (new_height, new_width*2), True)
    anoms = []
    raw_anoms = []
    x_vals = []
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #total = total//10
    frame_repeats = 0
    frame_repeat_start_idx = total//1.1
    frame_skip = 6
    with progressbar.ProgressBar(max_value=total+frame_repeats, widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                                           progressbar.ETA()]) as bar:
        bar.update(0)
        while success:
            # Encode --------------------------------------------------------------------
            frame = cv2.resize(orig_frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            frame = frame
            frame = force_divisible(frame, sp_grid_size)
            frame = (frame > 200) * 255
            frame = frame.astype(np.uint8)
            encoded_input = (frame == 255)[:, :, 0].astype(np.uint8)
            # Run HTM -------------------------------------------------------------------
            anom, colored_sp_output, raw_anom = grid_htm(encoded_input)
            anoms.append(anom)
            raw_anoms.append(raw_anom)
            x_vals.append(bar.value)
            # Create output -------------------------------------------------------------
            frame_out = np.zeros(shape=(frame.shape[0]*2, frame.shape[1], 3), dtype=np.uint8)
            colored_sp_output = cv2.resize(colored_sp_output, dsize=(scaled_sdr_shape[1], scaled_sdr_shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            frame_out[frame.shape[0]:frame.shape[0]+scaled_sdr_shape[0], 0:, :] = frame
            frame_out[0: frame.shape[0], 0:] = colored_sp_output
            frame_number = utils.text_phantom(str(bar.value), 12)
            frame_out[0:12, -(12 * 5):] = frame_number
            out.write(frame_out)

            # Get next frame -------------------------------------------------------------
            # Do not get next frame if it is currently set to be repeating the same frame
            for i in range(frame_skip):
                if bar.value < frame_repeat_start_idx or bar.value > frame_repeat_start_idx+frame_repeats:

                    success, orig_frame = vidcap.read()
                    orig_frame = concat_seg(orig_frame, success)

                bar.update(bar.value + 1)
                if bar.value == total:
                    break
            if bar.value == total:
                break

    dump_data = {"anom_scores": anoms, "raw_anoms": raw_anoms, "anom_markers": [31900, 35850, 102000, 117700, 135850, 148900], "x_vals": x_vals, "frame_freeze": frame_repeat_start_idx}
    pickle.dump(dump_data, open(f'surveillance_results/anoms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb'))
    pickle.dump(dump_data, open(f'surveillance_results/{args.name}.pkl', 'wb'))
