import jetson.inference
import jetson.utils

from imutils.video import FPS
import cv2
import time
from collections import OrderedDict
import math
import numpy as np 

from tracker import Tracker

def frame_add_crosshairs(frame,ID,x,y,r=20,lc=(0,0,255),cc=(0,0,255),lw=1,cw=1):

    x = int(round(x,0))
    y = int(round(y,0))
    r = int(round(r,0))

    cv2.line(frame,(x,y-r*2),(x,y+r*2),lc,lw)
    cv2.line(frame,(x-r*2,y),(x+r*2,y),lc,lw)

    cv2.circle(frame,(x,y),r,cc,cw)
    cv2.putText(frame,"ID[{}]".format(ID), (x+30,y), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0),1,cv2.LINE_AA,False)

camera = jetson.utils.gstCamera(640, 380, "0")

# init the trackers with a frame
img, width, height = camera.CaptureRGBA(zeroCopy=1)
tracker = Tracker(img)
targets = OrderedDict()
previousTargetID = None
targetID = None

while True:

    img, widthImg, heightImg = camera.CaptureRGBA(zeroCopy=1)
    targets = tracker.track(img, None, widthImg, heightImg)

    frame = jetson.utils.cudaToNumpy(img, width, height, 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    frame = frame / 255

    print(targets)
    for ID in targets:
        print(ID)
        frame_add_crosshairs(frame, ID, targets[ID][0][0], targets[ID][0][1])
    
    cv2.imshow("ID Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


    
