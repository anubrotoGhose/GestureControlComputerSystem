import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import pyautogui
###############################
wCam, hCam = 640, 480
###############################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cTime = 0
pTime = 0
c = 7
detector = htm.handDetector(detection_conf=0.7)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist)!=0:
        # print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[8][1], lmlist[8][2]

        x2, y2 = lmlist[20][1], lmlist[20][2]

        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        print(length)
        if length<40:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            print("Not Screenshot")
        else:
            print("Screenshot")
            image = pyautogui.screenshot()
            image = cv2.cvtColor(np.array(image), 
                     cv2.COLOR_RGB2BGR)
            cv2.imwrite("C:/Users/anubr/Python Projects/Machine Vision/Project/Screenshots/image"+str(c)+".png", image)
            c+=1
            time.sleep(3)

    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
    cv2.imshow("Img",img)
    cv2.waitKey(1)