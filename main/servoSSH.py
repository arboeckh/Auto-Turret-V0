import paramiko, time
import curses
import math

class SSHServo():
	
	def __init__(self):
		
		# Also need to geometric parameters used in calculateYawPitch
		self.pos = [math.pi/2, math.pi/2]	#initial positions of servos -> [yaw,pitch]
		self.centrePos = [math.pi/2,math.pi/2]	#used to store calibrated centres 

		self.ssh = paramiko.client.SSHClient()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.ssh.connect('bigspear', username='alphahawk', password='gloriousbattle')
		time.sleep(1)
		# Set position to middle
		self.cmd_servo('yaw',self.pos[0])
		self.cmd_servo('pitch',self.pos[1])
		time.sleep(1)

		

	def calibrateYAW_manually(self):

		#Define curses for calibration
		screen = curses.initscr()
		curses.noecho()
		curses.cbreak()
		screen.keypad(True)

		# calibrate yaw
		outputChanged = False

		try:
			while True:
				char = screen.getch()
				if char == ord('q'):
					break
				elif char == curses.KEY_RIGHT:
					self.pos[0] += 0.015
					screen.addstr(0, 0, 'right')
					outputChanged = True
				elif char == curses.KEY_LEFT:
					self.pos[0] -= 0.015
					screen.addstr(0, 0, 'left')
					outputChanged = True
				
				if self.pos[0] >= math.pi:
					self.pos[0] = math.pi - 0.015
				elif self.pos[0] <= 0:
					self.pos[0] = 0.015
				
				#convert output to bytes and send to arudino if changed
				if outputChanged:
					self.cmd_servo('yaw',self.pos[0])
					outputChanged = False
					

		finally:
			curses.nocbreak(); screen.keypad(0); curses.echo()
			curses.endwin()
			self.centrePos[0] = self.pos[0]	#set centre as current position

	def calibratePITCH_manually(self):

		#Define curses for calibration
		screen = curses.initscr()
		curses.noecho()
		curses.cbreak()
		screen.keypad(True)

		# calibrate yaw
		outputChanged = False

		try:
			while True:
				char = screen.getch()
				if char == ord('q'):
					break
				elif char == curses.KEY_RIGHT:
					self.pos[1] += 0.015
					screen.addstr(0, 0, 'right')
					outputChanged = True
				elif char == curses.KEY_LEFT:
					self.pos[1] -= 0.015
					screen.addstr(0, 0, 'left')
					outputChanged = True
				
				if self.pos[1] >= math.pi:
					self.pos[1] = math.pi - 0.015
				elif self.pos[1] <= 0:
					self.pos[1] = 0.015
				
				#convert output to bytes and send to arudino if changed
				if outputChanged:
					self.cmd_servo('pitch',self.pos[1])
					outputChanged = False
					

		finally:
			curses.nocbreak(); screen.keypad(0); curses.echo()
			curses.endwin()
			self.centrePos[1] = self.pos[1]	#set centre as current position

	
	def cmd_servo(self, axis, angle):
		
		dc_us = angle/math.pi * 1200 + 900
		if axis == 'yaw':
			ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command('pigs s 12 {}'.format(dc_us))
		elif axis == 'pitch':
			ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command('pigs s 13 {}'.format(dc_us))
		
	def set_position(self, yaw, pitch):
		# Add pitch
		#Yaw and pitch relative to the centrePos
		yaw += self.centrePos[0]
		pitch = self.centrePos[1] - pitch
		self.cmd_servo('yaw', yaw)
		self.cmd_servo('pitch', pitch)
	
	# def computeYawPitch(self, coordinates):

	
		




