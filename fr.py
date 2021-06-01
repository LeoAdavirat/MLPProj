

# CONFIG VALUES

Thick = 2 #THICKNESS OF BRUSH
DL = 10 #DEQUE LIMIT - LIMIT THE LENGTH OF RECENT LOCATIONS PRINT ON FRAMES (FOR CALCULATIONS)
RL = 5 #RENDER LIMIT - SAME AS DEQUE LIMIT BUT FOR VISUAL PURPOSE
detection_min_complexity = 0.5 #THIS ALTERS HARSHNESS OF FACE DETECTION FILTERS

# CODE DOWN HERE:

import cv2
import mediapipe as mp
import time
import collections
from collections import deque

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("카메라 찾는 유나 [ITZY].mp4")
cap = cv2.VideoCapture("Rosé is perfectly symmetrical [TikTok].mp4")
pTime = 0
PastLocations = deque(8 * DL * (0,) , maxlen = 8 * DL)


# MEDIAPIPE VARIABLES
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence = detection_min_complexity)


# for i in range(30):
while True:
	_, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
			REye, LEye, Nose, *_ = detection.location_data.relative_keypoints
			for part in [REye, LEye, Nose]:
				points = mpDraw._normalized_to_pixel_coordinates(part.x, part.y, FaceWidth, FaceHeight)
				PastLocations.appendleft(points[0])
				PastLocations.appendleft(points[1])
			for i in range(RL):
				cv2.circle(img, (PastLocations[i * 8 + 7], PastLocations[i * 8 + 6]), int(Thick / 2), (50,205,50), Thick * 3)
				cv2.circle(img, (PastLocations[i * 8 + 5], PastLocations[i * 8 + 4]), int(Thick / 2), (00,00,255), Thick * 2)
				cv2.circle(img, (PastLocations[i * 8 + 3], PastLocations[i * 8 + 2]), int(Thick / 2), (00,00,255), Thick * 2)
				cv2.circle(img, (PastLocations[i * 8 + 1], PastLocations[i * 8 + 0]), int(Thick / 2), (00,00,255), Thick * 2)
			cv2.rectangle(img, FaceBox, (255, 0, 255), Thick)
			cv2.putText(img, f'{int(detection.score[0] * 100)}%', (FaceBox[0], FaceBox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
			StableEyeX = sum(PastLocations[i * 8 + 5] for i in range(DL)) + sum(PastLocations[i * 8 + 3] for i in range(DL))
			StableEyeY = sum(PastLocations[i * 8 + 4] for i in range(DL)) + sum(PastLocations[i * 8 + 2] for i in range(DL))
			print(StableEyeX, StableEyeY)
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

	# SHOW FRAME
	cv2.imshow("img", img)
	cv2.waitKey(1)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()