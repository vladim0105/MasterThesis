import argparse
import cv2
import pathlib
import numpy as np


def video_to_dataset(in_file, out_file):
    vidcap = cv2.VideoCapture(in_file)
    success, frame = vidcap.read()

    count = 0
    dataset = {"frames": []}
    while success:
        print(f"Saving frame #{count}...")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[0:-100, 100:-100]
        frame = np.where(frame > 127, 1, 0)
        dataset["frames"].append(frame)
        success, frame = vidcap.read()
        count += 1
    np.save(out_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()
    video_to_dataset(args.in_file, args.out_file)
