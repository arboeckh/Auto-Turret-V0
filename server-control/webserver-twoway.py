from flask import Flask, Response, render_template
from flask_socketio import SocketIO

import jetson.inference
import jetson.utils

import argparse
import sys

import numpy as np 
import cv2

from flask import Flask, Response, render_template

import jetson.inference
import jetson.utils

import argparse
import sys

import numpy as np 
import cv2

def gstreamer_pipeline(
    capture_width=640,
    capture_height=380,
    display_width=640,
    display_height=380,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# Setup flask stuff
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# tell flask app to init socket on the app
socketio = SocketIO(app, ping_timeout=3, ping_interval=1)

class Classifier_Enable:
    def __init__(self):
         self.enable = False
    def set_enable(self, en): 
         self.enable = en

en_obj = Classifier_Enable()


@app.route('/')
def index():
    return render_template('index.html')

def gen(cam):
    while True:
        ret_val, img = cam.read()
        print(en_obj.enable)
        #if en_obj.enable:
            #detections = net.Detect(img, width, height, overlay=opt.overlay)
        frameJPG = cv2.imencode('.JPEG', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frameJPG + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cam),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('class_toggle')
def toggle_class(enable):
    en_obj.set_enable(enable)


if __name__ == "__main__":
    socketio.run(app)
