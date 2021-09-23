""" Module to implement all image processing functionality

INPUT: 	Next image frame
OUTPUT: Pairs of pixel values of tracked objects (both cameras)

Contains classes:
centroid:     Centroid ID tracking. Takes rectangles and updates IDs + returns centroid pairs
classifier:   Classify objects, has a few modes. All return rectangle and label
triagulator: Takes Centroid pair and returns 3D positions of objects

"""
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import math
import cv2

import jetson.inference
import jetson.utils
import argparse
import sys

class Centroid_Tracker():
	
	def __init__(self, stereo, maxDisappeared = 10):
		self.stereo = stereo
		self.nextObjectID = 0
		# key: object ID, value: centroid [(x,y),(x,y)] coordinates for left and right cameras
		if stereo:
			self.objectsR = OrderedDict()
		
		self.objectsL = OrderedDict()
		# if both disappeared for over 10 frames
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
	
	def register(self, centroid):
		if self.stereo:
			self.objectsR[self.nextObjectID] = [centroid,False]
		self.objectsL[self.nextObjectID] = [centroid, False]

		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1
	
	def deregister(self, objectID):
		if self.stereo:
			del self.objectsR[objectID]
		del self.objectsL[objectID]

		del self.disappeared[objectID]
	
	def update(self, inputCentroidsL, inputCentroidsR=None):
		##################################################################
		# Stereo update
		if self.stereo:
		
			#If both L and R have no detections, increment all disappeared ID values by 1
			if (len(inputCentroidsL) == 0) & (len(inputCentroidsR) == 0):
				for objectID in list(self.disappeared.keys()):
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
				return self.objectsL, self.objectsR
		
			
			
			if len(self.objectsL) == 0:
				for i in range(0, len(inputCentroidsL)):
					self.register(inputCentroidsL[i])

			else:
				if len(inputCentroidsL) > 0:
						
					
					objectIDs = list(self.objectsL.keys())
					objectCentroidsL = []
					for item, _ in list(self.objectsL.values()):
						objectCentroidsL.append(item)

					DL = dist.cdist(np.array(objectCentroidsL), inputCentroidsL)

					#find smallest value in row, then sort rows by min value
					rowsL = DL.min(axis=1).argsort()
					colsL = DL.argmin(axis=1)[rowsL]

					usedRowsL = set()
					usedColsL = set()

					#left camera
					for (row, col) in zip(rowsL, colsL):

						if row in usedRowsL or col in usedColsL:
							continue

						objectID = objectIDs[row]
						self.objectsL[objectID] = [inputCentroidsL[col], True]
						self.disappeared[objectID] = 0

						usedRowsL.add(row)
						usedColsL.add(col)

						unusedRowsL = set(range(0, DL.shape[0])).difference(usedRowsL)
						unusedColsL = set(range(0, DL.shape[1])).difference(usedColsL)


					#left
					if DL.shape[0] >= DL.shape[1]:

						for row in unusedRowsL:
							objectID = objectIDs[row]
							self.disappeared[objectID] += 1
							self.objectsL[objectID][1] = False

							if self.disappeared[objectID] > self.maxDisappeared:
								self.deregister(objectID)
					
					else:
						for col in unusedColsL:
							self.register(inputCentroidsL[col])
				else:
					for ID in list(self.objectsL.keys()):
						self.objectsL[ID][1] = False

				
		
			if len(self.objectsR) == 0:
				for i in range(0, len(inputCentroidsR)):
					self.register(inputCentroidsR[i])

			else:
				if len(inputCentroidsR) > 0:

					objectIDs = list(self.objectsR.keys())
					objectCentroidsR = []
					for item, _ in list(self.objectsR.values()):
						objectCentroidsR.append(item)

					DR = dist.cdist(np.array(objectCentroidsR), inputCentroidsR)

					#find smallest value in row, then sort rows by min value
					rowsR = DR.min(axis=1).argsort()
					colsR = DR.argmin(axis=1)[rowsR]
					usedRowsR = set()
					usedColsR = set()

					#right camera
					for (row, col) in zip(rowsR, colsR):

						if row in usedRowsR or col in usedColsR:
							continue

						objectID = objectIDs[row]
						self.objectsR[objectID] = [inputCentroidsR[col], True]
						self.disappeared[objectID] = 0

						usedRowsR.add(row)
						usedColsR.add(col)

						unusedRowsR = set(range(0, DR.shape[0])).difference(usedRowsR)
						unusedColsR = set(range(0, DR.shape[1])).difference(usedColsR)

					if DR.shape[0] >= DR.shape[1]:

						for row in unusedRowsR:
							objectID = objectIDs[row]
							self.disappeared[objectID] += 1
							self.objectsR[objectID][1] = False

							if self.disappeared[objectID] > self.maxDisappeared:
								self.deregister(objectID)
					
					else:
						for col in unusedColsR:
							self.register(inputCentroidsR[col])
				else:
					for ID in list(self.objectsR.keys()):
						self.objectsR[ID][1] = False
		
			return self.objectsL, self.objectsR
		
		######################################################################
		# Mono update
		else:

			if len(inputCentroidsL)==0:
				for objectID in list(self.disappeared.keys()):
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
				return self.objectsL
			
			if len(self.objectsL) == 0:
				for i in range(0, len(inputCentroidsL)):
					self.register(inputCentroidsL[i])
			
			else:
				if len(inputCentroidsL) > 0:
						
					
					objectIDs = list(self.objectsL.keys())
					objectCentroidsL = []
					for item, _ in list(self.objectsL.values()):
						objectCentroidsL.append(item)

					DL = dist.cdist(np.array(objectCentroidsL), inputCentroidsL)

					#find smallest value in row, then sort rows by min value
					rowsL = DL.min(axis=1).argsort()
					colsL = DL.argmin(axis=1)[rowsL]

					usedRowsL = set()
					usedColsL = set()

					#left camera
					for (row, col) in zip(rowsL, colsL):

						if row in usedRowsL or col in usedColsL:
							continue

						objectID = objectIDs[row]
						self.objectsL[objectID] = [inputCentroidsL[col], True]
						self.disappeared[objectID] = 0

						usedRowsL.add(row)
						usedColsL.add(col)

						unusedRowsL = set(range(0, DL.shape[0])).difference(usedRowsL)
						unusedColsL = set(range(0, DL.shape[1])).difference(usedColsL)


					#left
					if DL.shape[0] >= DL.shape[1]:

						for row in unusedRowsL:
							objectID = objectIDs[row]
							self.disappeared[objectID] += 1
							self.objectsL[objectID][1] = False

							if self.disappeared[objectID] > self.maxDisappeared:
								self.deregister(objectID)
					
					else:
						for col in unusedColsL:
							self.register(inputCentroidsL[col])
				else:
					for ID in list(self.objectsL.keys()):
						self.objectsL[ID][1] = False		
			return self.objectsL

class ClassifierFace():

	def __init__(self, frame, confidence=0.5):
		self.net = cv2.dnn.readNetFromCaffe("models/face/deploy.prototxt", "models/face/res10_300x300_ssd_iter_140000.caffemodel")
		self.confidence = confidence
		(self.H, self.W) = frame.shape[:2]

	def locate(self, frame):

		blob = cv2.dnn.blobFromImage(frame, 1.0, (self.H,self.W), (104.0, 177.0, 123.0) )
		self.net.setInput(blob)
		detections = self.net.forward()

		rects = []

		for i in range(detections.shape[2]) :
			if detections[0,0,i,2] > self.confidence:
				box = detections[0,0,i,3:7] * np.array( [self.W,self.H,self.W,self.H] )
				rects.append(box.astype("int"))
		
		return rects

class ClassifierHuman():

	def __init__(self, frame, confidence=0.5):
		self.net = cv2.dnn.readNetFromCaffe("models/classifier/MobileNetSSD_deploy.prototxt.txt", "models/classifier/MobileNetSSD_deploy.caffemodel")
		self.confidence = confidence
		(self.H, self.W) = frame.shape[:2]
		self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

	def locate(self, frame):
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,(300, 300), 127.5)
		self.net.setInput(blob)
		detections = self.net.forward()
		
		rects = []

		for i in range(detections.shape[2]):
			confidence = detections[0,0,i,2]
			if confidence > self.confidence:
				class_ind = int(detections[0,0,i,1])
				class_detected = self.classes[class_ind]
				if class_detected == "person":
					box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H]) 
					rects.append(box.astype("int"))
		
		return rects

class Triangulator():

	def __init__(self, cDist, hTheta, vTheta, imShape):
		self.cDist = cDist			# in m
		self.hTheta = hTheta		# in RAD
		self.vTheta = vTheta		# in RAD
		self.imWidth = imShape[1]	# pixels
		self.imHeight = imShape[0]	# pixels
		self.coordm2 = None
		self.coordm1 = None
		self.coord_avg = [0,0,0]

		#imaginary distance to frame in horizontal dimension in pixels
		self.focalDistHorizontal = (self.imWidth/2) / math.tan(hTheta/2)
		self.focalDistVertical = (self.imHeight/2) / math.tan(vTheta/2)

	def triangulate(self, centroids):
		#centroids[0] is (x,y) pixels of left camera, [1] for right
		# imShape (w,h) in pixels of image

		coordinates = [0,0,0]

		#First find angle form center (positive cw from center)
		thetaCentreLeftHorizontal = math.atan( (centroids[0][0]- self.imWidth/2) / self.focalDistHorizontal)
		thetaLeftHorizontal = math.pi/2 - thetaCentreLeftHorizontal
		# And for right side
		thetaCentreRightHorizontal = math.atan( (centroids[1][0]- self.imWidth/2) / self.focalDistHorizontal)
		thetaRightHorizontal = math.pi/2 + thetaCentreRightHorizontal

		#Find z
		coordinates[2] = self.cDist / ( 1/math.tan(thetaLeftHorizontal) + 1/math.tan(thetaRightHorizontal) )

		#find x from left camera alone
		coordinates[0] = coordinates[2] / math.tan(thetaLeftHorizontal) - self.cDist/2
		#coordinates[0] = coordinates[2] / math.tan(thetaLeftHorizontal) - self.cDist



		#find vertical angle with left camera
		thetaVertical = math.atan( (self.imHeight/2 - centroids[0][1]) / self.focalDistVertical)

		#find y position from this angle and z distance since this angle is the angle between then
		coordinates[1] = math.tan(thetaVertical) * coordinates[2] 

		# if self.coordm2 is None:
		# 	self.coordm2 = coordinates
		# 	return self.coordm2
		# elif self.coordm1 is None:
		# 	self.coordm1 = coordinates
		# 	return self.coordm1
		# else:
		# 	for i in range(3):
		# 		self.coord_avg[i] = ( self.coordm2[i] + self.coordm1[i] + coordinates[i] ) / 3 
		# 		self.coordm2[i] = self.coordm1[i]
		# 		self.coordm1[i] = coordinates[i]
		# 		return self.coord_avg


		return	coordinates 	# list length 3

class Tracker():
	#initialize tracker with frame and camera info. 
	def __init__(self, frame, cDist=None, hTheta=None, vTheta=None, target='human', nCamera = 2):
		self.nCamera = nCamera
		# init the centroid tracker class
		if nCamera == 1:
			self.centroid = Centroid_Tracker(stereo = False)
		elif nCamera == 2:
			self.centroid = Centroid_Tracker(stereo = True)
			self.triangulator = Triangulator(cDist, hTheta, vTheta, frame.shape[:2])
		else:
			raise ValueError('nCamera in Tracker must be 1 or 2')
		
		# init the SSD from jetson-inference
		if target is 'human':
			parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())
			parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
			parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
			parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
			parser.add_argument("--overlay", type=str, default="none", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
			parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

			is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

			try:
				self.opt = parser.parse_known_args()[0]
			except:
				print("")
				parser.print_help()
				sys.exit(0)

			# load the object detection network
			self.classifier = jetson.inference.detectNet(self.opt.network, sys.argv, self.opt.threshold)



	# for mono cam, use frameL
	def track(self, frameL, frameR, width, height):

		if self.nCamera == 1:
			detections = self.classifier.Detect(frameL, width, height, overlay=self.opt.overlay)
			centroids = []
			for detection in detections:
				if detection.ClassID == 1:
					centroid = [0]*2
					centroid[0] = detection.Center[0]
					centroid[1] = detection.Center[1]
					centroids.append(centroid)
			
			objects = self.centroid.update(centroids)
			return objects
		
		
		elif self.nCamera == 2:
			# Find face rects in L and R frames
			detectionsL = self.classifier.Detect(frameL, width, height, overlay=self.opt.overlay)
			centroidsL = []
			for detection in detectionsL:
				if detection.ClassID == 1:
					centroid = [0]*2
					centroid[0] = detection.Center[0]
					centroid[1] = detection.Center[1]
					centroidsL.append(centroid)

			detectionsR = self.classifier.Detect(frameR, width, height, overlay=self.opt.overlay)
			centroidsR = []
			for detection in detectionsR:
				if detection.ClassID == 1:
					centroid = [0]*2
					centroid[0] = detection.Center[0]
					centroid[1] = detection.Center[1]
					centroidsR.append(centroid)

			# Update centroid tracker and get centroid objects {'IDno': [x,y]}
			objectsL, objectsR = self.centroid.update(centroidsL, centroidsR)

			# Triangulation 
			targetableID = []
			for ID in list(objectsL.keys()):
				if objectsL[ID][1] & objectsR[ID][1]:
					targetableID.append(ID)

			targets = OrderedDict()
			
			for ID in targetableID:
				targets[ID][0] = self.triangulator.triangulate( [objectsL[ID][0], objectsR[ID][0]] )	#x,y,z
				targets[ID][1] = objectsL[ID][0]	# L pixel coordinates
				targets[ID][2] = objectsR[ID][0]	# R pixel coordinates

			return targets

