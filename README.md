# Auto-Turret-V0
This device aims to use a stereo camera to locate and ID targets in 3D sapce and point a laser at the desired target. 

## Already Implemented
1. Homemade 3D printed stereo camera using 2xRPI cameras
2. Implementation of SSD-MobileNetV2 on the jetson nano to locate humans
3. Multiple simultaneous target tracking by using the "last-closest" technique
4. Triangulation of targets to get 3D location

## To Do
The main limitation for continuing the project is hardware:
1. Image resolution is quite low (320 x 240) for neural net to run with 5fps on jetson nano -> need better GPU/accelerator
2. Laser pointer needs to have good open-loop precision -> need to design and machine a turret instead of using cheap servos

To Do List:

- [ ] Design precision turret
- [ ] Implement a Kalman filter for better target position estimation
- [ ] Create calibration protocols for the stereo camera and the turret system


