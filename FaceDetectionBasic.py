import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Rosé is perfectly symmetrical [TikTok].mp4")
# pTime = 

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
	success, img = cap.read()

	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = faceDetection.process(imgRGB)

	if results.detections:
		for id, detection in enumerate(results.detections):
			mpDraw.draw_detection(img, detection)
			# REyes, LEyes, NoseT, *_ = detection.location_data.relative_keypoints
			# REyes, LEyes, NoseT = mpDraw._normalized_to_pixel_coordinates()
			# REyes, LEyes, NoseT = (float(REyes[0].replace('x: ','')), float(REyes[1].replace('y: ', ''))), (float(LEyes[0].replace('x: ','')), float(LEyes[1].replace('y: ', ''))),\
			# 			(float(NoseT[0].replace('x: ','')), float(NosT[1].replace('y: ', '')))
			# cv2.circle(img, (REyes[0], REyes[1]), 1, (50,205,50), 4)
			# cv2.circle(img, (LEyes[0], LEyes[1]), 1, (50,205,50), 4)
			# cv2.circle(img, (NoseT[0], NoseT[1]), 1, (50,205,50), 4)
			# print(id, detection)
			# print(detection.score)
			# print(detection.location_data.relative_bounding_box)
			# bboxC = detection.location_data.relative_bounding_box
			# ih, iw, ic = img.shape
			# bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
			#        int(bboxC.width * iw), int(bboxC.height * ih)
			# print(bbox)
			# cv2.rectangle(img, bbox, (255, 0, 255), 2)
			# cv2.putText(img, f'{int(detection.score[0] * 100)}%',
			#             (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
			#             2, (255, 0, 255), 2)

	# cTime = time.time()
	# fps = 1 / (cTime - pTime)
	# pTime = cTime
	# cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
	# 			3, (0, 255, 0), 2)
	cv2.imshow("Image", img)
	cv2.waitKey(1)