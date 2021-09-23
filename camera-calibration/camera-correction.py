import numpy as np
import cv2
import glob
import yaml
from imutils.video import VideoStream
import time

cal_file = open("right-camera-calibration.yaml")

parsed_cal_file = yaml.load(cal_file, Loader=yaml.FullLoader)
mtx = np.array(parsed_cal_file.get('camera_matrix'))
# mtx = cv2.UMat(np.array(mtx, dtype=np.uint8))
dist = np.array(parsed_cal_file.get('dist_coeff'))
# dist = cv2.UMat(np.array(dist, dtype=np.uint8))


img = cv2.imread('frame.png')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# x,y,w,h = roi
# corrected_frame = dst[y:y+h, x:x+w]


cv2.imshow('before', img)
cv2.imshow('after',dst)
cv2.waitKey()
