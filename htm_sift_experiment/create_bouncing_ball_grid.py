import math
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import progressbar
from cv2 import cv2

import model
import utils


def fill_circle(cx, cy, r, arr) -> np.ndarray:
    x = np.arange(0, arr.shape[0])
    y = np.arange(0, arr.shape[1])
    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    arr[mask, :] = 255
    return arr


def create_video(path, r, gridhtm: model.GridHTM, shape):
    out = cv2.VideoWriter('bouncing_ball_grid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (shape[0], shape[1]), True)
    sp_out = cv2.VideoWriter('bouncing_ball_sp_grid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (shape[0]//2, shape[1]//2), True)
    anoms = []
    n_predicted_things_list = []
    # path = path[:18000]
    with progressbar.ProgressBar(max_value=len(path),
                                 widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                          progressbar.ETA()]) as bar:
        for point in path:
            frame = np.zeros(shape).astype(np.uint8)
            frame = fill_circle(point[0], point[1], r, frame)
            bit_frame = (frame == 255)[:, :, 0].astype(np.uint8)
            #sdr = model.numpy_to_sdr(bit_frame)
            anom, col_sp_output = gridhtm(bit_frame)
            anoms.append(anom)
            col = utils.value_to_hsv(anom, 0, 1)
            # Draw the borders
            borderWidth = 2
            frame[:borderWidth, :, :] = 255
            frame[:, :borderWidth, :] = 255
            frame[:, -borderWidth:, :] = 255
            frame[-borderWidth:, :, :] = 255
            # Apply HSV color
            frame[:, :, 0] = col[0]
            frame[:, :, 1] = col[1]

            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            out.write(frame)
            sp_out.write(col_sp_output)
            bar.update(bar.value + 1)
    return anoms, n_predicted_things_list


def bouncing_path(startx, starty, r, shape, num_bounces=20, speed=4):
    current_x = startx
    current_y = starty
    path = []
    current_bounce = 0
    g = 2
    yvel = 0
    xvel = 0
    print("Simulating path...")
    frame_id = 0
    start_xvel_frame = None
    stop_xvel_frame = None
    while current_bounce < num_bounces:

        if current_y+yvel >= shape[1] - r:
            yvel = -yvel
            current_bounce += 1
        else:
            yvel += g
        if current_bounce == num_bounces // 4 and xvel == 0:
            xvel = speed
            if start_xvel_frame is None:
                start_xvel_frame = frame_id
        if current_x+xvel > shape[1] - r:
            xvel = -speed
        if current_x+xvel < r:
            xvel = speed
        if current_bounce >= int(num_bounces * 0.8):
            xvel = 0
            if stop_xvel_frame is None:
                stop_xvel_frame = frame_id
        current_y += yvel
        current_x += xvel
        path.append([current_x, current_y])
        frame_id += 1
    anom_frames = [start_xvel_frame, stop_xvel_frame]
    return path, anom_frames


if __name__ == "__main__":
    frame_size = (120, 120, 3)
    sp_grid_size = 30
    tm_grid_size = 15
    sp_args = model.SpatialPoolerArgs()
    sp_args.inputDimensions = (sp_grid_size, sp_grid_size)
    sp_args.columnDimensions = (tm_grid_size, tm_grid_size)
    sp_args.potentialPct = 0.5
    sp_args.potentialRadius = 5
    sp_args.localAreaDensity = 0.1
    sp_args.globalInhibition = True
    sp_args.wrapAround = False
    sp_args.synPermActiveInc = 0.1
    sp_args.synPermInactiveDec = 0.001
    sp_args.stimulusThreshold = 1
    sp_args.boostStrength = 0
    sp_args.dutyCyclePeriod = 250

    tm_args = model.TemporalMemoryArgs()
    tm_args.columnDimensions = sp_args.columnDimensions
    tm_args.predictedSegmentDecrement = 0.0005
    tm_args.permanenceIncrement = 0.1
    tm_args.permanenceDecrement = 0.001
    tm_args.minThreshold = 3
    tm_args.activationThreshold = 5
    tm_args.cellsPerColumn = 16
    tm_args.seed = sp_args.seed
    r = 3
    A = int(math.pi * (r ** 2))
    grdhtm = model.GridHTM(frame_size, sp_grid_size, tm_grid_size, sp_args, tm_args, A)

    path, anom_frames = bouncing_path(frame_size[0] // 2, int(r * 1.3), r, shape=frame_size, num_bounces=1000)
    anom_scores, predicted_things = create_video(path, r, grdhtm, shape=frame_size)
    dump_data = {"anom_scores": anom_scores, "anom_markers": anom_frames, "predicted_things": predicted_things}
    pickle.dump(dump_data, open(f'bb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb'))
    pickle.dump(dump_data, open(f'bb_latest.pkl', 'wb'))
