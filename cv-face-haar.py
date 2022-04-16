import cv2
import time

cap = cv2.VideoCapture(0)

#Comment above and uncomment below 2 lines in case the web cam taking long time to start due to Windows Direct Draw
#cap = cv2.VideoCapture()
#cap.open(0, cv2.CAP_DSHOW)

cap.set(3, 1280)
cap.set(4, 720)

face_detector = cv2.CascadeClassifier("face.xml")
pTime = 0

while True:
    success, img = cap.read()
    if success:
        faces = face_detector.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow("Face Detection",img)

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
