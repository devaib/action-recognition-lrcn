import os
import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import time

filepath = "./data/human-action/video"
for full_root, dirs, files in os.walk(filepath):
    root = os.path.basename(full_root)
    start_time = time.time()
    if root not in ['boxing', 'handclapping', 'handwaving',
                    'jogging', 'running', 'walking']:
        continue
    for file in files:
        print(root + ': ' + file)
        videopath = os.path.join(filepath, root, file)
        cap = cv2.VideoCapture(videopath)
        videoname = os.path.splitext(file)[0]
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'data/human-action/frames', root, videoname)
        if not os.path.exists(path):
            os.makedirs(path)

        count = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            cv2.imshow('window-name',frame)
            imagename = root + "_" + videoname + "_frame%d.jpg" % count
            cv2.imwrite(os.path.join(path, imagename), frame)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        elapsed_time = time.time() - start_time
        m, s = divmod(elapsed_time, 60)
        h, m = divmod(m, 60)
        print "time: %d:%02d:%02d" % (h, m, s)