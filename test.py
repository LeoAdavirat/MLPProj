import cv2
import numpy as np

# inputfeed = "Untitleddsa Project.mp4"
inputfeed = "Rosé is perfectly symmetrical [TikTok].mp4"
# inputfeed = "카메라 찾는 유나 [ITZY].mp4"

cap = cv2.VideoCapture(inputfeed)

while True:
	_, frame = cap.read()
	cv2.imshow("frame", frame)
	cv2.waitKey(32)