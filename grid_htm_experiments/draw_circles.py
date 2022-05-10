import argparse

import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original_image", type=str)

    args = parser.parse_args()
    thickness = 15
    color = (28,134,238)
    img = cv2.imread(args.original_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.circle(img, (2083, 366), 100, color, thickness)
    cv2.circle(img, (2083, 366+450), 100, color, thickness)

    cv2.circle(img, (160, 1574), 100, color, thickness)
    cv2.circle(img, (160, 1574+450), 100, color, thickness)

    cv2.circle(img, (1115, 1574), 100, color, thickness)
    cv2.circle(img, (1115, 1574+450), 100, color, thickness)

    cv2.circle(img, (2083, 1574), 100, color, thickness)
    cv2.circle(img, (2083, 1574+450), 100, color, thickness)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.original_image, img)