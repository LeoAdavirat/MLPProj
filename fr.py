

# CONFIG VALUES

Thick = 2 #THICKNESS OF BRUSH
DL = 10 #DEQUE LIMIT - LIMIT THE LENGTH OF RECENT LOCATIONS PRINT ON FRAMES (FOR CALCULATIONS)
RL = 5 #RENDER LIMIT - SAME AS DEQUE LIMIT BUT FOR VISUAL PURPOSE
detection_min_complexity = 0.5 #THIS ALTERS HARSHNESS OF FACE DETECTION FILTERS

# CODE DOWN HERE:

import numpy as np
import cv2
import mediapipe as mp
import time
import collections
from collections import deque
from dataclasses import dataclass
from typing import Tuple


#VIDEO FEED
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("카메라 찾는 유나 [ITZY].mp4")
# cap = cv2.VideoCapture("Rosé is perfectly symmetrical [TikTok].mp4")
cap = cv2.VideoCapture("Untitleddsa Project.mp4")

#VARIABLE SET
pTime = 0
PastLocations = deque(8 * DL * (0,) , maxlen = 8 * DL)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence = detection_min_complexity)

@dataclass
class FaceDetection_er:
	Thickness: int
	DLC: int
	RLC: int
	ColorMode_1: Tuple[int, int, int] = (50, 205, 50)
	ColorMode_2: Tuple[int, int, int] = (00, 00, 255)

	def DrawPoints(self,
		img: np.ndarray, PastLocations: collections.deque):
		for i in range(self.RLC):
			cv2.circle(img, (PastLocations[i * 8 + 7], PastLocations[i * 8 + 6]), int(self.Thickness / 2), self.ColorMode_1, self.Thickness * 3)
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

FaceDetectionfr = FaceDetection_er(Thick, DL, RL)
f = 0
# for i in range(2000):
while True:
	_, img = cap.read()
	try:
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	except cv2.error:
		print('Error: No more frames detected or there are problems with input frames')
		break
	results = faceDetection.process(imgRGB)
	if results.detections:
		for id, detection in enumerate(results.detections):
			FaceBoxC = detection.location_data.relative_bounding_box
			KeyPoints = detection.location_data.relative_keypoints
			FaceHeight, FaceWidth, _ = img.shape	
			FaceBox = int(FaceBoxC.xmin * FaceWidth), int(FaceBoxC.ymin * FaceHeight), int(FaceBoxC.width * FaceWidth), int(FaceBoxC.height * FaceHeight)
			CenterX, CenterY = int(FaceBoxC.xmin * FaceWidth + FaceBoxC.width * FaceWidth / 2), int(FaceBoxC.ymin * FaceHeight + FaceBoxC.height * FaceHeight / 2)
			PastLocations.appendleft(CenterX)
			PastLocations.appendleft(CenterY)
			*key_px, _, _, _ = detection.location_data.relative_keypoints
			for part in key_px:
				points = mpDraw._normalized_to_pixel_coordinates(part.x, part.y, FaceWidth, FaceHeight)
				PastLocations.appendleft(points[0])
				PastLocations.appendleft(points[1])
			FaceDetectionfr.DrawPoints(img, PastLocations)
			cv2.rectangle(img, FaceBox, (255, 0, 255), Thick)
			cv2.putText(img, f'{int(detection.score[0] * 100)}%', (FaceBox[0], FaceBox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
			EX, EY, NX, NY, CEX, CEY = FaceDetectionfr.Stable(PastLocations)
			cv2.line(img, (EX, EY), (NX, NY), (50, 225, 50), Thick)
			cv2.line(img, (CEX, CEY), (NX, NY), (0, 0, 205), Thick)
			if f == 0:
				print('EX    EY   NX   NY   CEX   CEY')
				f += 1
			print('(' + '{:0>3}'.format(EX) + ', ' + '{:0>3}'.format(EY) + ')', '(' + '{:0>3}'.format(NX) + 
				',' + '{:0>3}'.format(NY) + ')', '(' + '{:0>3}'.format(CEX) + ', ' + '{:0>3}'.format(CEY) + ')')
	else:
		print('No faces found')
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
	# SHOW FRAME
	cv2.imshow("img", img)
	if cv2.waitKey(32) & 0xFF == ord(' '):
		break

cap.release()
cv2.destroyAllWindows()