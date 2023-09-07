import time
import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, smooth=True, segment=False, smoothSegment=True, detectionConf = 0.5, trackConf=0.5) -> None:
        self.mode = mode
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf


        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smooth, min_detection_confidence=self.detectionConf, min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255, 0, 0), cv2.FILLED)

        return lmList

    

if __name__ == "__main__":
    cap = cv2.VideoCapture('pose_estimation/videos/2.mp4')
    pTime = 0

    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img, draw=False)
        print(lmList[14])
        cv2.circle(img, (lmList[14][1],lmList[14][2]), 10, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        cv2.waitKey(10)