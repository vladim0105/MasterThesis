import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pixellib.torchbackend.instance import instanceSegmentation
def segs_to_video():
    out = cv2.VideoWriter("segvid.avi", cv2.VideoWriter_fourcc(*'XVID'), 24, (1280, 720),
                          False)
    frame_names = glob.glob('data/frames/*.jpeg.seg.npy')
    frame_names.sort()
    for frame_name in frame_names:
        print(frame_name)
        frame = np.load(frame_name)
        out.write(np.uint8(frame * 127))

if __name__ == "__main__":
    segs_to_video()
    assert False
    segmentation_engine = instanceSegmentation()
    segmentation_engine.load_model("data/pointrend_resnet101.pkl", network_backbone="resnet101")
    t = segmentation_engine.select_target_classes(person=True, car=True)

    frame_names = []
    for frame_name in glob.glob('data/frames/*.jpeg'):
        frame_names.append(frame_name)
    frame_names.sort()
    frame_skip = 5
    output_frames = []
    for i in range(0, len(frame_names), frame_skip):
        print(f"Processing {frame_names[i]}...")
        output_frame = np.zeros(shape=(720, 1280))
        segment_info, _ = segmentation_engine.segmentImage(frame_names[i], segment_target_classes=t)
        for obj_idx, class_name in enumerate(segment_info["class_names"]):
            if segment_info["scores"][obj_idx] < 60:
                continue
            pix_val = segment_info["class_ids"][obj_idx] + 1
            seg = segment_info["masks"][:, :, obj_idx] * pix_val
            output_frame = np.maximum(seg, output_frame)
        np.save(f"{frame_names[i]}.seg", output_frame)
