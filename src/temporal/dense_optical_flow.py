import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import numpy as np
import os

cap = cv2.VideoCapture("boxing.avi")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
count = 0

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'test')
if not os.path.exists(path):
    os.makedirs(path)

while(1):
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
    cv2.imwrite(os.path.join(path, '{}_{}.png'.format('boxing', count)), bgr)
    count = count + 1

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    prvs = next

cap.release()
cv2.destroyAllWindows()
