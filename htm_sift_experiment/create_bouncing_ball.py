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


def create_video(path, r, sp: model.SpatialPooler, tm: model.TemporalMemory, shape):
    out = cv2.VideoWriter('bouncing_ball.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (shape[0], shape[1]), True)
    sp_vidw = cv2.VideoWriter('bouncing_ball_sp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (shape[0]//2, shape[1]//2), False)
    anoms = []
    with progressbar.ProgressBar(max_value=len(path), widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                                           progressbar.ETA()]) as bar:
        for point in path:
            frame = np.zeros(shape).astype(np.uint8)
            frame = fill_circle(point[0], point[1], r, frame)
            bit_frame = (frame == 255)[:, :, 0].astype(np.uint8)
            sdr = model.numpy_to_sdr(bit_frame)
            sp_out = sp(sdr, True)
            pred, anom = tm(sp_out, True)
            anoms.append(anom)
            col = utils.value_to_hsv(anom, 0, 1)
            # Draw the borders
            frame[:1, :, :] = 255
            frame[:, :1, :] = 255
            frame[:, -1:, :] = 255
            frame[-1:, :, :] = 255
            # Apply HSV color
            frame[:, :, 0] = col[0]
            frame[:, :, 1] = col[1]

            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            out.write(frame)
            sp_vidw.write(model.sdr_to_numpy(sp_out).astype(np.uint8)*255)
            bar.update(bar.value+1)
    return anoms


def bouncing_path(startx, starty, r, shape, num_bounces=20):
    current_x = startx
    current_y = starty
    path = []
    current_bounce = 0
    g = 2
    yvel = 0
    xvel = 0
    print("Simulating path...")
    frame_id = 0
    _anom_frames = []
    while current_bounce < num_bounces:
        current_y += yvel
        current_x += xvel
        if current_y > shape[1] - r:
            yvel = -yvel
            current_bounce += 1
        else:
            yvel += g
            path.append([current_x, current_y])
        if current_bounce == num_bounces // 4 and xvel == 0:
            xvel = 4
        if current_x > shape[1] - r:
            xvel = -4
        if current_x < r:
            xvel = 4
        if current_bounce >= int(num_bounces * 0.8):
            xvel = 0
            _anom_frames.append(frame_id)
        frame_id+=1
    anom_frames = np.zeros(shape=(frame_id,))
    for frame_id in _anom_frames:
        anom_frames[frame_id] = 1
    return path, anom_frames


if __name__ == "__main__":
    frame_size = (120, 120, 3)
    sp_args = model.SpatialPoolerArgs()
    sp_args.inputDimensions = (frame_size[0], frame_size[1])
    sp_args.columnDimensions = (frame_size[0]//2, frame_size[1]//2)
    sp_args.potentialPct = 0.05
    sp_args.potentialRadius = 120
    sp_args.localAreaDensity = 0.05
    sp_args.globalInhibition = True
    sp_args.wrapAround = False
    sp_args.synPermActiveInc = 0.1
    sp_args.synPermInactiveDec = 0.05
    sp_args.stimulusThreshold = 6

    tm_args = model.TemporalMemoryArgs()
    tm_args.columnDimensions = sp_args.columnDimensions
    tm_args.predictedSegmentDecrement = 0.03
    tm_args.permanenceIncrement = 0.5
    tm_args.permanenceDecrement = 0.01
    tm_args.minThreshold = 11
    tm_args.activationThreshold = 13
    tm_args.cellsPerColumn = 16
    tm_args.seed = sp_args.seed

    sp = model.SpatialPooler(sp_args)
    tm = model.TemporalMemory(tm_args)


    r = 3
    path, anom_frames = bouncing_path(frame_size[0] // 2, int(r * 1.3), r, shape=frame_size, num_bounces=1000)
    anoms = create_video(path, r, sp, tm, shape=frame_size)
    pickle.dump(anoms, open(f'bb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb'))