import numpy as np
import progressbar
from cv2 import cv2

if __name__ == "__main__":
    video_scale = 0.3
    vidcap = cv2.VideoCapture('../data/sperm2.avi')
    success, frame = vidcap.read()
    out = cv2.VideoWriter('../data/sperm_seg2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (frame.shape[1], frame.shape[0]), False)
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    with progressbar.ProgressBar(max_value=total, widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                                           progressbar.ETA()]) as bar:
        while success:

            frame_out = (frame > 200)[:, :, 0] * 255
            frame_out = frame_out.astype(np.uint8)
            out.write(frame_out)
            success, frame = vidcap.read()
            bar.update(bar.value + 1)
            if bar.value == total:
                break
