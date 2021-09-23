from imutils.video import VideoStream
import imutils
import time
import cv2
from imutils.video import FPS
from collections import OrderedDict
import math
import numpy as np
import time

import display
from tracker import Tracker

import jetson.inference
import jetson.utils

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


# create video sources & outputs
# input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv, )
camera1 = jetson.utils.gstCamera(1920, 1080, "0")
camera2 = jetson.utils.gstCamera(1920, 1080, "1")

tracker = None
validTargets = OrderedDict()
previousTargetID = None
targetID = None

avg_factor = 3
coordinates = [0]*3
pixelL = [0]*2
pixelR = [0]*2

fps = FPS().start()

i=0
while i<100:
	# Get frames and resize
	img1, width, height = camera1.CaptureRGBA(zeroCopy=1)
	img2, width, height = camera2.CaptureRGBA(zeroCopy=1)

	# Init the tracker with the frame and camera data
	if tracker is None:
		tracker = Tracker(img1, 0.12, 1.3, 1.3, target='human')
	
	# Retreive target IDs, their 3D coordinates and pixel coordinates in both frames
	validTargets = tracker.track(img2, img1, width, height)

	# Target is first element in validTarget
	if len(validTargets) > 0:
		targetID = list(validTargets)[0]

		coordinates[0] = validTargets[targetID][0][0]
		coordinates[1] = validTargets[targetID][0][1]
		coordinates[2] = validTargets[targetID][0][2]
		pixelL[0] = validTargets[targetID][1][0]
		pixelL[1] = validTargets[targetID][1][1]
		pixelR[0] = validTargets[targetID][2][0]
		pixelR[1] = validTargets[targetID][2][1]

		print(coordinates)
	else:
		targetID = None
	i = i + 1




