# Usage
# python scripts/meanshift_tracking.py -u 'rtsp://admin:instar@192.168.2.19/livestream/13'
import cv2
# from matplotlib import pyplot as plt
import argparse
# from imutils import resize
from imutils.video import VideoStream

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", help="RTSP streaming URL", default="rtsp://admin:instar@192.168.2.19/livestream/13")
args = vars(ap.parse_args())

# get video stream from IP camera
print("[INFO] starting video stream")
vs = VideoStream(args["url"]).start()

# first frame from stream
frame = vs.read()
# optional - resize image if source too high res
# frame = resize(frame, width=1280)
# select region of interest
bbox = cv2.selectROI(frame)
x, y, w, h = bbox
track_window = (x, y, w, h)

# define area of bounding box as area of interest
roi = frame[y:y+h, x:x+w]
# [DEBUG] display region of interest
# cv2.imshow("ROI", roi)
# if cv2.waitKey(1) & 0xFF == ord('q'):
    # break

# convert frame to HSV colour space
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# get histogram for [0] blue, [1] green, [2] red channel
# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
# [DEBUG] display region of interest histogram
# plt.hist(roi.ravel(), 256, [0, 256])
# plt.show()
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# convert hist values 0-180 to a range between 0-1
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# set up the termination criteria, either 10 iteration or move by at least 1 pt
parameter = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# now loop through the rest of avail frames
# and use meanshift to track defined roi
while True:
    # get next frame
    frame = vs.read()
    if True:
        # convert to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # compare blue channel of current with roi histogram
        # https://docs.opencv.org/3.4.15/da/d7f/tutorial_back_projection.html
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # call meanshift() to find match of histogram in current frame
        # and get the new coordinates
        ok, track_window = cv2.meanShift(dst, (x, y, w, h), parameter)
        if not ok:
            print('[WARNING] track lost')
        # now update the roi coordinates to new values
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        # display track
        cv2.imshow("Meanshift Track", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cv2.destroyAllWindows()
