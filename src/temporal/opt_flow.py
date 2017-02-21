from __future__ import print_function
import numpy as np
import os
import sys; sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import video


def draw_flow(img, flow, step=4):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':

    actionname = 'boxing'
    path = os.path.dirname(os.path.realpath(__file__))
    path_test = os.path.join(path, actionname)
    path_horz = os.path.join(path_test, 'horizontal')
    path_vert = os.path.join(path_test, 'vertical')
    if not os.path.exists(path_horz):
        os.makedirs(path_horz)
    if not os.path.exists(path_vert):
        os.makedirs(path_vert)

    print(__doc__)
    # try:
    #     fn = sys.argv[1]
    # except IndexError:
    #     fn = 0

    fn = os.path.join(path, "boxing.avi")
    cam = video.create_capture(fn)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')
        cv2.imshow('Horizontal Component', horz)
        cv2.imwrite(os.path.join(path_horz, '{}_{}.jpg'.format(actionname, count)), horz)
        cv2.imshow('Vertical Component', vert)
        cv2.imwrite(os.path.join(path_vert, '{}_{}.jpg'.format(actionname, count)), vert)

        prevgray = gray
        count = count + 1

        cv2.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])

    cv2.destroyAllWindows()
