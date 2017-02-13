import os
import cv2

cap = cv2.VideoCapture('data/test.avi')
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'data/frames')
if not os.path.exists(path):
    os.makedirs(path)

count = 0
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('window-name',frame)
    imagename = "frame%d.jpg" % count
    cv2.imwrite(os.path.join(path, imagename), frame)
    count = count + 1
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
