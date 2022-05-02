import numpy as np
import progressbar
from cv2 import cv2
from matplotlib import pyplot as plt

from sperm_experiment import concat_seg

if __name__ == "__main__":
    video_scale = 0.3
    vidcap = cv2.VideoCapture('../data/output_seg_2.mp4')
    success, frame = vidcap.read()
    frame = concat_seg(frame, success)
    scaled_frame_shape = (int(frame.shape[0] * video_scale), int(frame.shape[1] * video_scale))
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    cell_sizes = [12]
    empty_pattern_pixels = 0
    min_sparsity = 0
    num_active_pixels_list = {cell_size:[] for cell_size in cell_sizes}
    idx = 0
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
                if num_active_pixels <= min_sparsity:
                    num_active_pixels = empty_pattern_pixels

                num_active_pixels_list[cell_size].append(num_active_pixels)
            success, frame = vidcap.read()
            idx += 1
            bar.update(idx)

            if bar.value == total:
                break
    plt.rc('font', size=16)
    plt.rc('axes', titlesize=20)
    for cell_size in cell_sizes:
        array = np.array(num_active_pixels_list[cell_size])
        plt.title(f"$\sigma={np.std(array):.2f}$")
        non_zero_avg = np.mean(array[np.nonzero(array)])
        plt.hist(num_active_pixels_list[cell_size], bins="auto", log=True, alpha=1)
        plt.axvline(non_zero_avg, c="red", label="Non-zero Mean")
    plt.legend(loc='upper left')
    plt.ylabel("Frames")
    plt.xlabel("Number of Active Pixels")
    plt.tight_layout()
    plt.savefig("active_pixels_dist.eps")
    plt.show()
