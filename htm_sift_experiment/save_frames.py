import argparse
from pathlib import Path

from cv2 import cv2
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.font_manager

#matplotlib.rc('text', usetex=True)
fpath = Path("LUCON.TTF")
plt.rcParams["figure.figsize"] = (4.8*1.2, 4.8)
def get_frames(path, pos, amount, frame_frame_skip):
    frames = []
    vid = cv2.VideoCapture(path)
    vid.set(1, pos)
    for i in range(0, amount*frame_frame_skip, frame_frame_skip):
        ret, still = vid.read()
        still = cv2.cvtColor(still, cv2.COLOR_BGR2RGB)
        frames.append(still)
        vid.set(1, pos+i)
    return frames


def show_images(images, rows=1, cols=1, frame_frame_skip=1, anomaly_idx=-1):
    figure, ax = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True)
    for idx, image in enumerate(images):
        ax.ravel()[idx].imshow(image)
        ax.ravel()[idx].set_axis_off()
        text = f"t+{idx*frame_frame_skip}"
        color ="#000000"
        if idx == 0:
            text = "t"
        if idx == anomaly_idx:
            color="#FF0000"
        ax.ravel()[idx].text(0.5, -0.1, f"{text}", size=10, ha="center",
                             transform=ax.ravel()[idx].transAxes, font=fpath, color=color)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=.001)
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=str)
    argparser.add_argument("frame", type=int)
    argparser.add_argument("-n","--amount", type=int, default=6)
    argparser.add_argument("-c","--cols", type=int, default=3)
    argparser.add_argument("-r","--rows", type=int, default=2)
    argparser.add_argument("-s","--skip", type=int, default=8)
    argparser.add_argument("-m","--mark", type=int, default=-1)

    args = argparser.parse_args()
    assert args.cols*args.rows == args.amount, "cols*rows must match the amount of frames!"
    plt.rcParams["figure.figsize"] = (4.8 * args.cols/2.5, 4.8)
    frames = get_frames(args.file, args.frame//6, args.amount, frame_frame_skip=args.skip)
    show_images(frames, args.rows, args.cols, frame_frame_skip=args.skip, anomaly_idx=args.mark)
