# Face Detection with Haar Cascade
<img src="https://user-images.githubusercontent.com/61585411/167343360-8475ff3a-b3f1-43a3-ac0b-40ddadecd232.jpg" width=600>

While we can obtain significantly higher accuracy and more robust face detections with deep learning face detectors, OpenCV’s Haar cascades still have their place:
- They are lightweight
- They are super fast, even on resource-constrained devices
- The Haar cascade model size is tiny (930 KB)
Yes, there are several problems with Haar cascades, namely that they are prone to false-positive detections and less accurate than their HOG + Linear SVM, SSD, YOLO, etc., counterparts. However, they are still useful and practical, especially on resource-constrained devices.

## Procedures
1. Import the libraries.
2. Setting up a webcam.
3. Creating face detector
4. Do face detection by using Haar Cascade
5. Displaying the output

## Step 1: Import the libraries
```python
import cv2
import time
```
## Step 2: Setting up a webcam (Windows)
```python
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
```
It is quicker to get web cam live in Windows environment by adding cv2.CAP_DSHOW attribute.
## Step 2: Setting up a webcam (Windows/Linux/Mac)
```python
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
```
## Step 3: Initialize the face detector by using CascadeClassifier
```python
face_detector = cv2.CascadeClassifier("face.xml")
```
## Step 4: Do face detection by using Haar Cascade
```python
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
```
## Step 5: Displaying the output
```python
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow("Face Detection",img)

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
```
## References
- [OpenCV Face detection with Haar cascades](https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/)
- [Face Detection with HAAR Cascade in OpenCV Python](https://machinelearningknowledge.ai/face-detection-with-haar-cascade-in-opencv-python/)
