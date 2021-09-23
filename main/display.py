""" Collection of display functions """

import cv2

def printCoordinates(frame, coordinates, centroid, velocity):
	if velocity is None:
		text = 'X: {:2.2f}\nY: {:2.2f}\nZ: {:2.2f}\n'.format(coordinates[0],coordinates[1],coordinates[2])
		frame_add_crosshairs(frame, centroid[0], centroid[1])
	else:
		text = 'X: {:2.2f}, {:2.2f}\nY: {:2.2f}, {:2.2f}\nZ: {:2.2f}, {:2.2f}\n'.format(coordinates[0],velocity[0],coordinates[1],velocity[1],coordinates[2],velocity[2])
		frame_add_crosshairs(frame, centroid[0], centroid[1])
	# cv2.putText(frame, text, (centroid[0] - 20, centroid[1] - 20),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	lineloc = 0
	lineheight = 30
	for t in text.split('\n'):
		lineloc += lineheight
		cv2.putText(frame,
					t,
					(centroid[0]+30,centroid[1]- 2*lineheight + lineloc), # location
					cv2.FONT_HERSHEY_PLAIN, # font
					#cv2.FONT_HERSHEY_SIMPLEX, # font
					1.5, # size
					(0,255,0), # color
					1, # line width
					cv2.LINE_AA, #
					False) #


def frame_add_crosshairs(frame,x,y,r=20,lc=(0,0,255),cc=(0,0,255),lw=1,cw=1):

	x = int(round(x,0))
	y = int(round(y,0))
	r = int(round(r,0))

	cv2.line(frame,(x,y-r*2),(x,y+r*2),lc,lw)
	cv2.line(frame,(x-r*2,y),(x+r*2,y),lc,lw)

	cv2.circle(frame,(x,y),r,cc,cw)

def targeting_info(frameL, frameR, coordinates, pixelL, pixelR, currentVelocity=None):

	#text = "ID {}".format(ID)
	text = "Target"
	cv2.putText(frameL, text, (pixelL[0] - 10, pixelL[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.putText(frameR, text, (pixelR[0] - 10, pixelR[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	velocity = currentVelocity
	printCoordinates(frameL, coordinates, pixelL, velocity)
	printCoordinates(frameR, coordinates, pixelR, velocity)

