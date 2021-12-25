import cv2.cv2 as cv2

if __name__ == "__main__":
    orig_vidcap = cv2.VideoCapture('../data/parking01.mp4')
    seg_vidcap = cv2.VideoCapture('../data/output_seg.mp4')
    seg2_vidcap = cv2.VideoCapture('../data/output_seg_2.mp4')

    _, orig_frame = orig_vidcap.read()
    _, seg_frame = seg_vidcap.read()
    _, seg2_frame = seg2_vidcap.read()

    cv2.imwrite("orig_frame.png", orig_frame)
    cv2.imwrite("seg_frame.png", seg_frame)
    cv2.imwrite("seg2_frame.png", seg2_frame)
