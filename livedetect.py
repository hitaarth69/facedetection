import cv2




face= "C:/Users/ADMIN/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
eye = "C:/Users/ADMIN/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_eye.xml"

face_classifier = cv2.CascadeClassifier(face)
eye_classifier = cv2.CascadeClassifier(eye)

vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, 1.1,5)
    eyes = eye_classifier.detectMultiScale(grey, 1.1,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),4)

    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

    cv2.imshow("window", frame )

    cv2.waitKey(0) 

vid.release()
cv2.destroyAllWindows()    

