import cv2
import mediapipe as mp
import dataclasses
# MEDIAPIPE VARIABLES


@dataclasses.dataclass
class FaceRecognizerModule:
	def GetData():
		mpFaceDetection = mp.solutions.face_detection
		mpDraw = mp.solutions.drawing_utils
		faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence = 0.5)

	def ProcessImage(img):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convert Image froom RBG to RGB
		FaceBoxC = detection.location_data.relative_bounding_box
		FaceHeight, FaceWidth, Facec = img.shape
		FaceBox = int(FaceBoxC.xmin * FaceWidth), int(FaceBoxC.ymin * FaceHeight), int(FaceBoxC.width * FaceWidth), int(FaceBoxC.height * FaceHeight)
		CenterX, CenterY = int(FaceBoxC.xmin * FaceWidth + FaceBoxC.width * FaceWidth / 2), int(FaceBoxC.ymin * FaceHeight + FaceBoxC.height * FaceHeight / 2)
		FacePoints = detection.location_data.relative_keypoints
		print(FacePoints)
		return CenterX, CenterY