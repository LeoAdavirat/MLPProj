import MainLibrary
# import TestLibrary
import numpy as np
import cv2


#CONFIG VALUES

Thick = 2 #THICKNESS OF BRUSH
DL = 10 #DEQUE LIMIT - LIMIT THE LENGTH OF RECENT LOCATIONS PRINT ON FRAMES (FOR CALCULATIONS)
RL = 5 #RENDER LIMIT - SAME AS DEQUE LIMIT BUT FOR VISUAL PURPOSE
detection_min_complexity = 0.5 #THIS ALTERS HARSHNESS OF FACE DETECTION FILTERS
FPS_display = True
MultiplyingFrequency = 15 #THIS DISPLAYS MORE POINTS ON THE GRAPHS
sensitivity = 16 #HOW SENSITIVE THE DETECTOR IS
# sensitivity = int(input())

# inputfeed = "Untitleddsa Project.mp4"
# inputfeed = "Rosé is perfectly symmetrical [TikTok].mp4"
# inputfeed = "카메라 찾는 유나 [ITZY].mp4"

# FaceDetectionER = TestLibrary.FaceDetection_er(Thick, DL, RL, input_feed = inputfeed)
FaceDetectionER = MainLibrary.FaceDetection_er(Thick, DL, RL, sensitivity)
FaceDetectionER.InitVars()
checker = None
currentiter = None
while True:
	FaceDetectionER.IterFrame()
	currentiter = FaceDetectionER.Translate()
	if checker != currentiter:
		checker = currentiter
		if checker != 0:
			print(checker)