import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import os
import cv2

debug = True

videopath = os.path.join('./boxing.avi')
cap = cv2.VideoCapture(videopath)

prevgray = None

count = 0
while cap.isOpened():
    count += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)

    # generate optical flow
    if prevgray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prevgray = gray
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')


cap.release()

