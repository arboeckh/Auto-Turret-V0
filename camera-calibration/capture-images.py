from imutils.video import VideoStream

import numpy as np 
import cv2
import time

vs = VideoStream(src="nvarguscamerasrc ! video/x-raw(memory:NVMM), " \
	"width=(int)1280, height=(int)720,format=(string)NV12, " \
	"framerate=(fraction)120/1 ! nvvidconv ! video/x-raw, " \
	"format=(string)BGRx ! videoconvert ! video/x-raw, " \
	"format=(string)BGR ! appsink").start()
time.sleep(2.0)

frame = vs.read()
frame = cv2.flip(frame, 0)

cv2.imwrite('frame.png', frame)


vs.stop()
cv2.destroyAllWindows()





    
    
