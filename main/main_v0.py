""" This is the main auto-turret script 

TO DO: Each camera capture and classifier in own thread, assess fps improvement

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
from kinematics import Velocity_Estimate, calculateYawPitch
from servoSSH import SSHServo

servos = SSHServo()
servos.calibrateYAW_manually()
servos.calibratePITCH_manually()

vsL = VideoStream(src=1).start()
vsR = VideoStream(src=0).start()
time.sleep(2.0)

servoParams = [0.070, -0.025, 0.015, 0.037]

tracker = None
validTargets = OrderedDict()
previousTargetID = None
targetID = None
curentVelocity = None

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
			velocityEstimator = Velocity_Estimate(validTargets[targetID][0],time.time())
			currentVelocity = None
			previousTargetID = targetID
			print("Tracking new target")
		else:
			currentVelocity = velocityEstimator.currentVelocity(validTargets[targetID][0], time.time())
		
		yaw, pitch = calculateYawPitch(validTargets[targetID][0], servoParams)
		servos.set_position(yaw, pitch)
	else:
		currentVelocity = None

	


	# Display coordinate and ID information on frames
	display.targeting_info(frameL, frameR, validTargets, targetID, currentVelocity)
	frameS = np.hstack( (frameL, frameR) )
	cv2.imshow("Stereo", frameS)

	# Exit if q is pressed
	key = cv2.waitKey(1) & 0XFF
	if key == ord('q'):
		break

cv2.destroyAllWindows()
vsL.stop()
vsR.stop()
