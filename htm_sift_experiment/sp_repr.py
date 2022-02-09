import cv2
import numpy as np

import model

if __name__ == "__main__":
    frame = cv2.imread("../Thesis/resources/methodology/car_segmentation.png", cv2.IMREAD_REDUCED_GRAYSCALE_2)
    frame = (frame > 200) * 255
    frame = frame.astype(np.uint8)

    sp_args = model.SpatialPoolerArgs()
    sp_args.seed = 2
    sp_args.inputDimensions = (frame.shape[0], frame.shape[1])
    sp_args.columnDimensions = (frame.shape[0]//3, frame.shape[1]//3)
    sp_args.potentialRadius = 5
    sp_args.potentialPct = 0.01
    sp_args.localAreaDensity = 0.05
    sp_args.globalInhibition = True

    print("aaaa")
    frame = model.numpy_to_sdr(frame)
    sp = model.SpatialPooler(sp_args)
    print("aaaa")
    out = sp(frame, False)
    out = model.sdr_to_numpy(out)*255

    cv2.imwrite("sp_repr.png", out)

