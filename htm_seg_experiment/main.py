import glob

import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from pixellib.torchbackend.instance import instanceSegmentation


def segment_info_to_frame(segment_info, frame_shape, classes):
    output_frame = np.zeros(shape=(frame_shape[0]*len(classes), frame_shape[1]), dtype=np.uint8)
    for obj_idx, class_name in enumerate(segment_info["class_names"]):
        if segment_info["scores"][obj_idx] < 40:
            continue
        #pix_val = segment_info["class_ids"][obj_idx] + 1
        pix_val = 255
        seg = segment_info["masks"][:, :, obj_idx] * pix_val
        class_id = classes.index(class_name)
        output_frame[frame_shape[0]*class_id:frame_shape[0]*(class_id+1)][:] = np.maximum(seg, output_frame[frame_shape[0]*class_id:frame_shape[0]*(class_id+1)][:])
    return output_frame


if __name__ == "__main__":
    segmentation_engine = instanceSegmentation()
    segmentation_engine.load_model("../data/pointrend_resnet101.pkl", network_backbone="resnet101")
    t = segmentation_engine.select_target_classes(person=True, car=True)

    vidcap = cv2.VideoCapture('../data/parking01.mp4')

    success, frame = vidcap.read()
    classes = ["car", "person"]
    out = cv2.VideoWriter('../data/output_seg_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (frame.shape[1], frame.shape[0]*len(classes)), 0)
    frame_id = 0
    while success:
        print(f"Processing frame #{frame_id}")
        segment_info, _ = segmentation_engine.segmentFrame(frame, segment_target_classes=t)
        output_frame = segment_info_to_frame(segment_info, (frame.shape[0], frame.shape[1]), classes).astype(np.uint8)
        if frame_id == 2:
            cv2.imwrite("../data/test.png", output_frame)
        out.write(output_frame)
        success, frame = vidcap.read()
        frame_id += 1
