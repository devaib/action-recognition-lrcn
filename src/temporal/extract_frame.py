import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import os
import time
import cv2
import re
from annotation import Anno, Annos
from opt_flow import *

debug = False

p_num = re.compile('\d+')   # pattern

anno_path = os.path.join("../data/human-action", "00sequences.txt")
# list of annotations
annos = Annos()
with open(anno_path) as txtfile:
    for line in txtfile:
        if line.split() == [] or line.split()[1] == '*missing*':
            continue
        for idx, phrase in enumerate(line.split()):
            frames = {}
            if idx == 0:
                for inx, word in enumerate(phrase.split('_')):
                    if inx == 0:
                        m_num = p_num.findall(word) # match
                        num = m_num[0]
                    if inx == 1:
                        actionname = word
                    if inx == 2:
                        condition = word
                anno = Anno(num, actionname, condition)
                annos.add_anno(anno)

            if idx > 1:
                m_num = p_num.findall(phrase)   # match
                num0 = int(m_num[0])
                num1 = int(m_num[1])
                anno.add_frames([num0, num1])
        if idx < 5:
            anno.add_frames([0, 0]) # padding for person with less action


filepath = "../data/human-action/video"
start_time = time.time()
for full_root, dirs, files in os.walk(filepath):
    root = os.path.basename(full_root)
    if root not in ['boxing', 'handclapping', 'handwaving',
                    'jogging', 'running', 'walking']:
        continue
    for file in files:
        print(root + ': ' + file)
        videopath = os.path.join(filepath, root, file)
        cap = cv2.VideoCapture(videopath)
        videoname = os.path.splitext(file)[0]

        # fetch annotation
        anno = annos.get_anno_by_videoname(videoname)
        frames = anno.get_frames()  # list of 4 annotations
        person_num = anno.get_person_num()
        condition = anno.get_condition()
        if 11 <= int(person_num) <= 18:
            setname = 'train'
        elif 19 <= int(person_num) <= 25 or int(person_num) in [1, 4]:
            setname = 'val'
        else:
            setname = 'test'

        actionname = root
        # frame path
        path0 = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path0, '../data/human-action/frames', setname,
                            actionname, 'person'+person_num, condition)

        # optical flow path
        of_path = os.path.join(path0, '../data/human-action/optical-flow', setname,
                               actionname, 'person'+person_num, condition)
        prevgray = None

        count = 0
        while cap.isOpened():
            count += 1
            if debug:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            ret,frame = cap.read()
            if not ret:
                break
            # check frames validation
            if frames[0][0] <= count <= frames[0][1]:
                sub = '1'
            elif frames[1][0] <= count <= frames[1][1]:
                sub = '2'
            elif frames[2][0] <= count <= frames[2][1]:
                sub = '3'
            elif frames[3][0] <= count <= frames[3][1]:
                sub = '4'
            else:
                prevgray = None
                continue

            # extract frames
            if debug:
                cv2.imshow('frame', frame)
            imagename = actionname + "_" + videoname + "_frame%d.jpg" % count
            fullpath = os.path.join(path, sub)
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
            cv2.imwrite(os.path.join(fullpath, imagename), frame)

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
            path_horz = os.path.join(of_path, sub, 'horizontal')
            path_vert = os.path.join(of_path, sub, 'vertical')
            if not os.path.exists(path_horz):
                os.makedirs(path_horz)
            if not os.path.exists(path_vert):
                os.makedirs(path_vert)
            cv2.imwrite(os.path.join(path_horz,
                                     '{}_{}_frame{}.jpg'.format(actionname, videoname, count)), horz)
            cv2.imwrite(os.path.join(path_vert,
                                     '{}_{}_frame{}.jpg'.format(actionname, videoname, count)), vert)
            if debug:
                cv2.imshow('Vertical Component', vert)
                cv2.imshow('Horizontal Component', horz)
                cv2.imshow('flow', draw_flow(gray, flow))
                cv2.imshow('flow HSV', draw_hsv(flow))

        cap.release()
        elapsed_time = time.time() - start_time
        m, s = divmod(elapsed_time, 60)
        h, m = divmod(m, 60)
        print "time: %d:%02d:%02d" % (h, m, s)


