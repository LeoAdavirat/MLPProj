

# CONFIG DEFAULT VALUES

Thick = 2 #THICKNESS OF BRUSH
DL = 50 #DEQUE LIMIT - LIMIT THE LENGTH OF RECENT LOCATIONS PRINT ON FRAMES (FOR CALCULATIONS)
RL = 5 #RENDER LIMIT - SAME AS DEQUE LIMIT BUT FOR VISUAL PURPOSE
detection_min_complexity = 0.5 #THIS ALTERS HARSHNESS OF FACE DETECTION FILTERS
MultiplyingFrequency = 15 #THIS DISPLAYS MORE POINTS ON THE GRAPHS
sensitivity = 16 #HOW SENSITIVE THE DETECTOR IS

# CODE DOWN HERE:

import numpy as np
import cv2
import mediapipe as mp
import time
import collections
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import Tuple
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import statistics

#VIDEO FEED
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("카메라 찾는 유나 [ITZY].mp4")
# cap = cv2.VideoCapture("Rosé is perfectly symmetrical [TikTok].mp4")
# cap = cv2.VideoCapture("Untitleddsa Project.mp4")

#VARIABLE SET


@dataclass
class FaceDetection_er:
	Thickness: int = Thick
	DLC: int = DL
	RLC: int = RL
	sensitivity: float = sensitivity
	_MultiplyingFrequency: int = MultiplyingFrequency
	input_feed: str or int = 0
	ColorMode_1: Tuple[int, int, int] = (50, 205, 50)
	ColorMode_2: Tuple[int, int, int] = (00, 00, 255)
	b = 3

	def InitVars(self):
		self.cap = cv2.VideoCapture(self.input_feed)
		self.pTime = 0
		self.PastLocations = np.zeros((3, self.DLC), np.uint8)
		self.mpFaceDetection = mp.solutions.face_detection
		self.mpDraw = mp.solutions.drawing_utils
		self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence = detection_min_complexity)
		self.FaceDetectionfr = FaceDetection_er()
		self.PhuongSai = deque(maxlen = 3 * (self.DLC - 1))
		self.SumOfPS = deque(maxlen = 3 * self.DLC)
		self.FaceWidth = None
		self.id = False
		# self.BestLine = 
		# for i in range(2000):

	def IterFrame(self):
		_, img = self.cap.read()
		# debugimg = np.zeros(img.shape)
		try:
			imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		except cv2.error:
			raise FrameNotFoundError('Error: no more frames detected or there are problems with input frames')
		results = self.faceDetection.process(imgRGB)
		if results.detections:
			for id, detection in enumerate(results.detections):
				FaceHeight, self.FaceWidth, _ = img.shape			
				*key_px, _, _, _ = detection.location_data.relative_keypoints
				temp = []
				for part in key_px:
					points = self.mpDraw._normalized_to_pixel_coordinates(part.x, part.y, self.FaceWidth, FaceHeight)
					temp.append([points[0]])
				# print(temp)
				self.PastLocations = np.append(self.PastLocations, temp, axis = 1)
				self.PastLocations = np.delete(self.PastLocations, slice(None, 1), axis = 1)
				cv2.line(img, (self.PastLocations[0][-1], 0), (self.PastLocations[0][-1], FaceHeight), (50, 205, 50), 3)
				cv2.line(img, (self.PastLocations[1][-1], 0), (self.PastLocations[1][-1], FaceHeight), (205, 50, 205), 3)
				cv2.line(img, (self.PastLocations[2][-1], 0), (self.PastLocations[2][-1], FaceHeight), (125, 125, 125), 3)
		cTime = time.time()
		self.fps = 1/(cTime-self.pTime)
		self.pTime = cTime
		cv2.putText(img, f'FPS: {int(self.fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
		# SHOW FRAME
		cv2.imshow("img", img)
		if cv2.waitKey(32) & 0xFF == ord(' '):
			cap.release()
			cv2.destroyAllWindows()
			raise KeyBoardInterruptError("Pressing Space has interrupted the program")

	def Translate(self):
		if self.FaceWidth != None:
			Ave = [statistics.fmean(self.PastLocations[i]) for i in range(3)]
			check = []
			for i in range(3):
				tem = self.PastLocations[i][-1] - Ave[i]
				if tem <= -self.sensitivity:
					check.append(-1)
				if tem >= self.sensitivity:
					check.append(1)
				if -self.sensitivity < tem < self.sensitivity:
					check.append(0)
			return sum(check)
		else:
			print('No Faces Detected')

	# def Translate(self):
	# 	self.PhuongSai = [self.PastLocations[i + 4] - self.PastLocations[i] for i in range(self.DLC - 1)]
	# 	for i in range(3):
	# 		self.SumOfPS.appendleft([self.PhuongSai[j + i] for j in range(self.DLC - 1)])
	# 	print([self.SumOfPS[i] for i in range(3)])
	# def CalculateTrend(self):
	# 	CL = linear_model.LinearRegression()
	# 	model = CL.fit([PastLocations[i] for i in range(0, self.DLC, 3)], [PastLocations[i + 1] for i in range(0, self.DLC, 3)], [PastLocations[i + 2] for i in range(0, self.DLC, 3)])
# FaceDetectionER = FaceDetection_er(Thick, DL, RL, input_feed = inputfeed)
# FaceDetectionER.MainProgram()