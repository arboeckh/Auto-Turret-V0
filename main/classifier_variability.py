"""
This is to test how consistent are the centroids of the bounding rectangles generated
by the classifier. 

"""

from imutils.video import VideoStream
import imutils
import time
import cv2
from collections import OrderedDict
import math
import numpy as np
import time

import display
from tracker import Tracker

vsL = VideoStream(src=2).start()
vsR = VideoStream(src=0).start()

# Sleep for 10 seconds to get into place
time.sleep(1)


tracker = None
validTargets = OrderedDict()
previousTargetID = None
targetID = None

targetLocations = np.zeros( (200,3) )
i = 0

while True:
	# Get frames and resize
	frameL = vsL.read()
	frameR = vsR.read()
	frameL = imutils.resize(frameL, width=400)
	frameR = imutils.resize(frameR, width=400)

	# Init the tracker with the frame and camera data
	if tracker is None:
		tracker = Tracker(frameL, 0.12, 1.3, 1.3)
	
	# Retreive target IDs, their 3D coordinates and pixel coordinates in both frames
	validTargets = tracker.track(frameL, frameR)

	# Target is first element in validTarget
	if len(validTargets) > 0:
		targetID = list(validTargets)[0]

		if targetID is not previousTargetID:
			previousTargetID = targetID
			print("Tracking new target")
	else:
		targetID = None
		

	

	if targetID is not None:
	# Display coordinate and ID information on frames
		display.targeting_info(frameL, frameR, validTargets, targetID)
		frameS = np.hstack( (frameL, frameR) )
		cv2.imshow("Stereo", frameS)

		targetLocations[i] = validTargets[targetID][0]
		i += 1
		if i == 200:
			break

	# Exit if q is pressed
	key = cv2.waitKey(1) & 0XFF
	if key == ord('q'):
		break

cv2.destroyAllWindows()
vsL.stop()
vsR.stop()

minX = np.min(targetLocations[:,0])
maxX = np.max(targetLocations[:,0])
meanX = np.mean(targetLocations[:,0])

minY = np.min(targetLocations[:,1])
maxY = np.max(targetLocations[:,1])
meanY = np.mean(targetLocations[:,1])

minZ = np.min(targetLocations[:,2])
maxZ = np.max(targetLocations[:,2])
meanZ = np.mean(targetLocations[:,2])

print("x info")
print(minX)
print(maxX)
print(meanX)
print("y info")
print(minY)
print(maxY)
print(meanY)
print("z info")
print(minZ)
print(maxZ)
print(meanZ)


