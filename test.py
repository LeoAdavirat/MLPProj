import cv2
import numpy as np
import time

# inputfeed = "Untitleddsa Project.mp4"
inputfeed = "Rosé is perfectly symmetrical [TikTok].mp4"
# inputfeed = "카메라 찾는 유나 [ITZY].mp4"

cap = cv2.VideoCapture(inputfeed)
pTime = 0
while True:
	_, frame = cap.read()
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
	cv2.imshow("frame", frame)
	cv2.waitKey(32)