# Usage
# python scripts/optical_flow_sparse_manual.py -p 'resources/car_race_01.mp4'
import datetime
import sys
import numpy as np
import cv2
import argparse

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="Path to video file", default="resources/car_race_02.mp4")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["path"])
if not cap.isOpened():
    print("[ERROR] cannot open video file")
    sys.exit()

# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS |
                                                                      cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (100, 3))

# get first video frame
ok, frame = cap.read()
if not ok:
    print("[ERROR] cannot get frame from video")
    sys.exit()
# convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Use optical flow to detect object corners / edges from initial frame
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask = None, **parameters_shitomasi)
# [Debug] show amount of found edges
# max value = maxCorners see above. Reduce qualityLevel to get more hits
# print(len(edges))

# create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)

# Optional recording parameter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
video_codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
prefix = 'recordings/'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")
basename = "object_track.mp4"
video_output = cv2.VideoWriter("_".join([prefix, basename]), video_codec, fps, (frame_width, frame_height))

# loop through the remaining frames of the video
# and apply algorithm to track selected objects
while True:
    # get next frame
    ok, frame = cap.read()
    if not ok:
        print("[INFO] end of file reached")
        break
    # prepare grayscale image
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # update object corners by comparing with found edges in initial frame
    update_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None,
                                                         **parameter_lucas_kanade)
    # only update edges if algorithm successfully tracked
    new_edges = update_edges[status == 1]
    # to calculate directional flow we need to compare with previous position
    old_edges = edges[status == 1]

    for i, (new, old) in enumerate(zip(new_edges, old_edges)):
        a, b = new.ravel()
        c, d = old.ravel()

        # draw line between old and new corner point with random colour
        mask = cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), colours[i].tolist(), 2)
        # draw circle around new position
        frame = cv2.circle(frame, (int(a), int(b)), 5, colours[i].tolist(), -1)

    result = cv2.add(frame, mask)
    # optional recording result/mask
    # video_output.write(result)
    cv2.imshow('Optical Flow (sparse)', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # overwrite initial frame with current before restarting the loop
    frame_gray_init = frame_gray.copy()
    # update to new edges before restarting the loop
    edges = new_edges.reshape(-1, 1, 2)


cv2.destroyAllWindows()
cap.release()