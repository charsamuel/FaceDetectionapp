import xml.etree.ElementTree as ET
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.3,minNeighbors=5)
  for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
  return img  
cap = cv2.VideoCapture(0)  

while True:
   ret, frame = cap.read()
   frame = detect_faces(frame)
   
   cv2.imshow('Video Face Detection',frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):
     break
   
cap.release()  
cv2.destroyAllWindows() 