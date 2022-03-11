from cv2 import cv2
from matplotlib import pyplot as plt
import matplotlib

#matplotlib.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (4.8*2, 4.8)
font = {'family' : 'normal',
        'weight' : 'normal',
        "style": "italic",
        'size'   : 12}
matplotlib.rc('font', **font)
def get_frames(path, pos, amount):
    frames = []
    vid = cv2.VideoCapture(path)
    vid.set(1, pos)
    for i in range(amount):
        ret, still = vid.read()
        still = cv2.cvtColor(still, cv2.COLOR_BGR2RGB)
        frames.append(still)
    return frames


def show_images(images, rows=1, cols=1):
    figure, ax = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True)
    for idx, image in enumerate(images):
        ax.ravel()[idx].imshow(image)
        ax.ravel()[idx].set_axis_off()
        text = f"t+{idx}"
        if idx == 0:
            text = "t"
        ax.ravel()[idx].text(0.5, -0.1, f"{text}", size=12, ha="center",
                             transform=ax.ravel()[idx].transAxes)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=.001)
    plt.show()


if __name__ == "__main__":
    frames = get_frames("surveillance_results/output.mp4", 1000, 8)
    show_images(frames, 2, 4)
