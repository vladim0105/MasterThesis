import numpy as np
import progressbar
from cv2 import cv2
from matplotlib import pyplot as plt

from sperm_experiment import concat_seg

if __name__ == "__main__":
    video_scale = 0.3
    vidcap = cv2.VideoCapture('../data/sperm_seg2.mp4')
    success, frame = vidcap.read()
    frame = concat_seg(frame, success)
    scaled_frame_shape = (int(frame.shape[0] * video_scale), int(frame.shape[1] * video_scale))
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))//10
    cell_sizes = [16]
    empty_pattern_pixels = 0
    num_active_pixels_list = {cell_size:[] for cell_size in cell_sizes}

    with progressbar.ProgressBar(max_value=total, widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                                           progressbar.ETA()]) as bar:
        while success:
            frame = cv2.resize(frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            for cell_size in cell_sizes:
                cell = frame[
                       scaled_frame_shape[0] // 2 - cell_size // 2:scaled_frame_shape[0] // 2 + cell_size // 2,
                       scaled_frame_shape[1] // 2 - cell_size // 2:scaled_frame_shape[1] // 2 + cell_size // 2
                       ]
                num_active_pixels = (cell == 255)[:, :, 0].sum()
                if num_active_pixels == 0:
                    num_active_pixels = empty_pattern_pixels

                num_active_pixels_list[cell_size].append(num_active_pixels)
            success, frame = vidcap.read()
            bar.update(bar.value + 1)
            if bar.value == total:
                break
    #num_active_pixels_list = np.array(num_active_pixels_list)
    #print(np.std(num_active_pixels_list))
    #non_zero_avg = np.median(num_active_pixels_list[np.nonzero(num_active_pixels_list)])
    for cell_size in cell_sizes:
        plt.hist(num_active_pixels_list[cell_size], bins="auto", log=True, alpha=0.3)
    #plt.axvline(non_zero_avg, c="red", label="Non-zero Median")
    plt.legend()
    plt.ylabel("Frames")
    plt.xlabel("Number of Active Pixels")
    plt.show()