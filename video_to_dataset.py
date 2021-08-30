import argparse
import cv2
import pathlib
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    vidcap = cv2.VideoCapture(args.file)
    success, frame = vidcap.read()

    count = 0
    dataset = {"frames": []}
    while success:
        print(f"Saving frame #{count}...")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dataset["frames"].append(frame)
        success, frame = vidcap.read()
        count += 1
    np.save(args.name, dataset)

