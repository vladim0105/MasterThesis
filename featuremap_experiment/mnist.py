import numpy as np
import struct
import matplotlib.pyplot as plt
from cv2 import cv2

with open('../data/t10k-images-idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))
    for i in range(3):
        original = data[i, :, :]
        _, threshold = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"original_{i}.png", original)
        cv2.imwrite(f"threshold_{i}.png", threshold)
    plt.imshow(data[0, :, :], cmap='gray')
    plt.show()
