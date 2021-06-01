import cv2

cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	cv2.rectangle(frame, (0,0), (50, 50), (0,0,255), 2)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()