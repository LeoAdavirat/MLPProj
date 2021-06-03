

# CONFIG DEFAULT VALUES

Thick = 2 #THICKNESS OF BRUSH
DL = 10 #DEQUE LIMIT - LIMIT THE LENGTH OF RECENT LOCATIONS PRINT ON FRAMES (FOR CALCULATIONS)
RL = 5 #RENDER LIMIT - SAME AS DEQUE LIMIT BUT FOR VISUAL PURPOSE
detection_min_complexity = 0.5 #THIS ALTERS HARSHNESS OF FACE DETECTION FILTERS
MultiplyingFrequency = 15 #THIS DISPLAYS MORE POINTS ON THE GRAPHS

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
from matplotlib.animation import FuncAnimation
import math

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
	_MultiplyingFrequency: int = MultiplyingFrequency
	input_feed: str or int = 0
	ColorMode_1: Tuple[int, int, int] = (50, 205, 50)
	ColorMode_2: Tuple[int, int, int] = (00, 00, 255)
	b = 3

	def DrawPoints(self,
		img: np.ndarray, PastLocations: collections.deque):
		for i in range(self.RLC):
			cv2.circle(img, (PastLocations[i * 8 + 5], PastLocations[i * 8 + 4]), int(self.Thickness / 2), self.ColorMode_2, self.Thickness * 2)
			cv2.circle(img, (PastLocations[i * 8 + 3], PastLocations[i * 8 + 2]), int(self.Thickness / 2), self.ColorMode_2, self.Thickness * 2)
			cv2.circle(img, (PastLocations[i * 8 + 1], PastLocations[i * 8 + 0]), int(self.Thickness / 2), self.ColorMode_2, self.Thickness * 2)

	def Stable(self, PastLocations: collections.deque) -> Tuple[int, int, int, int]:
		StableEyeX = sum(PastLocations[i * 8 + 5] for i in range(self.DLC)) + sum(PastLocations[i * 8 + 3] for i in range(self.DLC))
		StableEyeY = sum(PastLocations[i * 8 + 4] for i in range(self.DLC)) + sum(PastLocations[i * 8 + 2] for i in range(self.DLC))
		StableEyeX, StableEyeY = (int(i / (2 * self.DLC)) for i in [StableEyeX, StableEyeY])
		StableNose = int(sum(PastLocations[i * 8 + 1] for i in range(self.DLC)) / self.DLC), int(sum(PastLocations[i * 8 + 0] for i in range(self.DLC)) / self.DLC)
		CurrentEyeX = int((PastLocations[5] + PastLocations[3]) / 2)
		CurrentEyeY = int((PastLocations[4] + PastLocations[2]) / 2)
		return StableEyeX, StableEyeY, StableNose[0], StableNose[1], CurrentEyeX, CurrentEyeY

	# def Translate(self):
	# 	self.Ave.appendleft(sum(PastLocations[i + 5] for i in range(1, self.DLC)) / (self.DLC - 1))
	# 	self.Ave.appendleft(sum(PastLocations[i + 4] for i in range(1, self.DLC)) / (self.DLC - 1))
	# 	self.Ave.appendleft(sum(PastLocations[i + 3] for i in range(1, self.DLC)) / (self.DLC - 1))
	# 	self.Ave.appendleft(sum(PastLocations[i + 2] for i in range(1, self.DLC)) / (self.DLC - 1))
	# 	self.Ave.appendleft(sum(PastLocations[i + 1] for i in range(1, self.DLC)) / (self.DLC - 1))
	# 	self.Ave.appendleft(sum(PastLocations[i + 0] for i in range(1, self.DLC)) / (self.DLC - 1))
	# 	self.TrendR.appendleft(math.sqrt((PastLocations[5] - self.Ave[5]) ** 2 + (PastLocations[4] - self.Ave[4]) ** 2))
	# 	self.TrendR.appendleft(math.sqrt((PastLocations[3] - self.Ave[3]) ** 2 + (PastLocations[2] - self.Ave[2]) ** 2))
	# 	self.TrendR.appendleft(math.sqrt((PastLocations[1] - self.Ave[1]) ** 2 + (PastLocations[0] - self.Ave[0]) ** 2))
	# 	self.TrendR.appendleft(union())


	def InitVars(self):
		self.cap = cv2.VideoCapture(self.input_feed)
		self.pTime = 0
		self.PastLocations = deque(6 * self.DLC * (0,) , maxlen = 6 * self.DLC)
		self.mpFaceDetection = mp.solutions.face_detection
		self.mpDraw = mp.solutions.drawing_utils
		self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence = detection_min_complexity)
		self.FaceDetectionfr = FaceDetection_er(self.Thickness, self.DLC, self.RLC)
		self.TrendR = deque(maxlen = 2 * (self.DLC - 1))
		self.TrendL = deque(maxlen = 2 * (self.DLC - 1))
		self.TrendN = deque(maxlen = 2 * (self.DLC - 1))
		self.Ave = deque(maxlen = 6)
		self.Trend = deque(maxlen = self.DLC - 1)
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
				FaceHeight, FaceWidth, _ = img.shape
				*key_px, _, _, _ = detection.location_data.relative_keypoints
				for part in key_px:
					points = self.mpDraw._normalized_to_pixel_coordinates(part.x, part.y, FaceWidth, FaceHeight)
					self.PastLocations.appendleft(points[0])
					self.PastLocations.appendleft(points[1])
				self.FaceDetectionfr.DrawPoints(img, self.PastLocations)
				print(tuple(self.PastLocations[i] for i in range(6)) + (FaceHeight, FaceWidth))
		else:
			print('No faces found')
		cTime = time.time()
		self.fps = 1/(cTime-self.pTime)
		self.pTime = cTime
		cv2.putText(img, f'FPS: {int(self.fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
		# SHOW FRAME
		# cv2.imshow("img", img)
		if cv2.waitKey(32) & 0xFF == ord(' '):
			cap.release()
			cv2.destroyAllWindows()
			raise KeyBoardInterruptError("Pressing Space has interrupted the program")

	def CalculateTrend():



# FaceDetectionER = FaceDetection_er(Thick, DL, RL, input_feed = inputfeed)
# FaceDetectionER.MainProgram()