import numpy as np
import cv2
import glob
import yaml
from imutils.video import VideoStream
import time

print("Camera R:0 or L:1")
camera = int(input())
assert (camera==1 or camera==0)

vs = VideoStream(src="nvarguscamerasrc ! video/x-raw(memory:NVMM), " \
	"width=(int)1280, height=(int)720,format=(string)NV12, " \
	"framerate=(fraction)120/1 ! nvvidconv ! video/x-raw, " \
	"format=(string)BGRx ! videoconvert ! video/x-raw, " \
	"format=(string)BGR ! appsink").start()
time.sleep(2.0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# store object and image points from all images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

goodCapt = 0
broken = False
while goodCapt < 30 :
    
    frame = vs.read()
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # if found, append object and image points
    if ret == True:
        objpoints.append(objp)
        # reffine the pixels
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # draw on frame
        img = cv2.drawChessboardCorners(frame, (9,6), corners2, ret)
        cv2.imshow('img', frame)
        goodCapt += 1
        key = cv2.waitKey(500) & 0xFF
        if key == ord("q"):
            broken = True
            break
    
    else:
        cv2.imshow('img', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            broken = True
            break

vs.stop()
cv2.destroyAllWindows()

if broken == False:
    # perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # transform the matrix and distortion coefficients to writable lists
    data = {'camera_matrix': np.asarray(mtx).tolist(),
            'dist_coeff': np.asarray(dist).tolist()}

    if camera == 0:
        fileName = 'right-camera-calibration'
    else:
        fileName = 'left-camera-calibration'
    with open(fileName+".yaml", "w") as f:
        yaml.dump(data, f)
    


else:
    print("ABORTED CALIBRATION")

