import time
import cv2
import mediapipe as mp
from PoseEstimationModule import PoseDetector



cap = cv2.VideoCapture('pose_estimation/videos/2.mp4')
pTime = 0

detector = PoseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img, draw=False)
    # if lmList:
        # cv2.circle(img, (lmList[14][1],lmList[14][2]), 10, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(10)