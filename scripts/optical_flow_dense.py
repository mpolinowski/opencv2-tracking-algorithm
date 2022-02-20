# Usage
# python scripts/optical_flow_sparse_manual.py -p 'resources/car_race_01.mp4'
import datetime
import sys

import cv2
import argparse
import numpy as np

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="Path to video file", default="resources/car_race_02.mp4")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["path"])
if not cap.isOpened():
    print("[ERROR] opening video file")
    sys.exit()

# Optional recording parameter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
video_codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
prefix = 'recordings/'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")
basename = "object_track.mp4"
video_output = cv2.VideoWriter("_".join([prefix, basename]), video_codec, fps, (frame_width, frame_height))

ok, first_frame = cap.read()
if not ok:
    print("[ERROR] getting frame from video")
    sys.exit()
frame_gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# create canvas to paint on
hsv_canvas = np.zeros_like(first_frame)
# set saturation value (position 2 in HSV space) to 255
hsv_canvas[..., 1] = 255

while True:
    # get next frame
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] reached end of file")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # compare initial frame with current frame
    flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    # get x and y coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # set hue of HSV canvas (position 1)
    hsv_canvas[..., 0] = angle*(180/(np.pi/2))
    # set pixel intensity value (position 3
    hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)

    # optional recording result/mask
    video_output.write(frame_rgb)

    cv2.imshow('Optical Flow (dense)', frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # set initial frame to current frame
    frame_gray_init = frame_gray

cv2.destroyAllWindows()
cap.release()