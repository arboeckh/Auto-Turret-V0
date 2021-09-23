import time 
import serial

stepsPerRevolution = 200*32

print(b'640')

try:
	ser = serial.Serial('COM12', 9600)
	print('com12')
except:
	ser = serial.Serial('COM11', 9600)
	print('com11')

ser.isOpen()	#checks that serial is open?

print('Serial is open')

time.sleep(1)


# steps = int(0.1*stepsPerRevolution)

# command = str(steps) + '\n'
# command_utf8 = command.encode('utf-8')

log = ser.write(b'640\n')

print(log)

print('Sent Serial CMD')


