import argparse
import cv2
import pathlib
import numpy as np

    
def dataset_to_video(dataset, out_file):

    frames = dataset["frames"]
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'XVID'), 24, (frames[0].shape[1], frames[0].shape[0]), False)
    for frame in frames:
        out.write(np.uint8(frame*255))
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()
    dataset = np.load(args.in_file, allow_pickle=True).item()
    dataset_to_video(args.in_file, args.out_file)


