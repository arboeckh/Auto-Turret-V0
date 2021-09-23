""" Takes in raw position values and computes acceleration, velocity, etc. 

INPUT:  3D position
OUTPUT: Depends on calculations 

"""
import math

class Velocity_Estimate():
	def __init__(self, initPosition, timeStamp):
		self.lastPosition = initPosition
		self.lastTimeStamp = timeStamp
		self.velocity = [0]*6

	def currentVelocity(self, position, timeStamp):
		for i in range(len(position)):
			self.velocity[i] = (position[i] - self.lastPosition[i]) / (timeStamp - self.lastTimeStamp)
		self.lastPosition = position
		self.lastTimeStamp = timeStamp
		return self.velocity


def calculateYawPitch(coordinates, params):
	# params in form [yUp, zIn, nzIn, radius]
	
	#apply offset to yaw axis
	x = coordinates[0]
	y = coordinates[1] - params[0]
	z = coordinates[2] - params[1]

	# Yaw is combined x and z
	yaw = math.atan(x/z)
	# New x is resultant of x and z
	nx = math.sqrt(x**2 + z**2)

	# Now offset new x value
	nx = nx - params[2]
	ny = y

	# quadratic equation params
	a = ny**2 + nx**2
	b = -2 * params[3]**2 * ny 
	c = params[3]**4 - params[3]**2 * nx**2

	# find 2 solution
	yc = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
	if yc < 0 :
		yc = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
		assert(yc > 0)
	
	#means pitch is upwards, so x must be negative
	if (ny - yc) >= 0:
		xc = -math.sqrt(params[3]**2 - yc**2)
	else:
		xc = math.sqrt(params[3]**2 - yc**2)
	
	# now find pitch angle, relative to standard 0 angle
	# POSSIBLE DIVIDE BY ZERO ERROR WHEN Y IS EXACTLY SAME LEVEL AS LASE AT 0 DEGREES, NEED TO FIX
	pitch = -math.atan(yc/xc) + math.pi/2  #to make it relative to vertical

	return yaw, pitch



# params = [0.070, -0.025, 0.015, 0.037]
# coordinates = [0,0.07+0.036,2]

# yaw, pitch = calculateYawPitch(coordinates, params)

