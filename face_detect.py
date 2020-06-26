import cv2 as cv
# press 'q' to quit
# choose the right path of the classifier file
faceCascade = cv.CascadeClassifier('/home/rawan/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)


while True:
	success, img = cap.read()
	faces = faceCascade.detectMultiScale(img,1.1,4)
	for (x,y,w,h) in faces:
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	cv.imshow("video",img)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()

