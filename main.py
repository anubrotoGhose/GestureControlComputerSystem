import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import pyautogui
from pyautogui import FailSafeException
import mediapipe as mp
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# Used to convert protobuf message to a dictionary. 
from google.protobuf.json_format import MessageToDict
###############################
wCam, hCam = 640, 480
###############################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cTime = 0
pTime = 0
c = 7

screen_width, screen_height = pyautogui.size()
print(pyautogui.size())

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange() # (-63.5, 0.0, 0.5)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initializing the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
	static_image_mode=False, 
	model_complexity=1, 
	min_detection_confidence=0.75, 
	min_tracking_confidence=0.75, 
	max_num_hands=2) 

def thumbs_down(lmlist):
    x0, y0 = lmlist[0][1], lmlist[0][2]
    x1, y1 = lmlist[1][1], lmlist[1][2]
    x2, y2 = lmlist[2][1], lmlist[2][2]
    x3, y3 = lmlist[3][1], lmlist[3][2]
    x4, y4 = lmlist[4][1], lmlist[4][2]
    if y4 > y3 > y2 > y1 > y0:
        return True
    return False

def thumbs_up(lmlist):
    x0, y0 = lmlist[0][1], lmlist[0][2]
    x1, y1 = lmlist[1][1], lmlist[1][2]
    x2, y2 = lmlist[2][1], lmlist[2][2]
    x3, y3 = lmlist[3][1], lmlist[3][2]
    x4, y4 = lmlist[4][1], lmlist[4][2]
    if y4 < y3 < y2 < y1 < y0:
        return True
    return False


def one(lmlist):
    x0, y0 = lmlist[0][1], lmlist[0][2]
    x5, y5 = lmlist[5][1], lmlist[5][2]
    x6, y6 = lmlist[6][1], lmlist[6][2]
    x7, y7 = lmlist[7][1], lmlist[7][2]
    x8, y8 = lmlist[8][1], lmlist[8][2]
    if y8 < y7 < y6 < y5 < y0:
        return True
    return False

def two(lmlist):
    x0, y0 = lmlist[0][1], lmlist[0][2]
    x9, y9 = lmlist[9][1], lmlist[9][2]
    x10, y10 = lmlist[10][1], lmlist[10][2]
    x11, y11 = lmlist[11][1], lmlist[11][2]
    x12, y12 = lmlist[12][1], lmlist[12][2]
    if y12 < y11 < y10 < y9 < y0:
        return True
    return False

def three(lmlist):
    x0, y0 = lmlist[0][1], lmlist[0][2]
    x13, y13 = lmlist[13][1], lmlist[13][2]
    x14, y14 = lmlist[14][1], lmlist[14][2]
    x15, y15 = lmlist[15][1], lmlist[15][2]
    x16, y16 = lmlist[16][1], lmlist[16][2]
    if y16 < y15 < y14 < y13 < y0:
        return True
    return False


def four(lmlist):
    x0, y0 = lmlist[0][1], lmlist[0][2]
    x17, y17 = lmlist[17][1], lmlist[17][2]
    x18, y18 = lmlist[18][1], lmlist[18][2]
    x19, y19 = lmlist[19][1], lmlist[19][2]
    x20, y20 = lmlist[20][1], lmlist[20][2]
    if y20 < y19 < y18 < y17 < y0:
        return True
    return False


def five(lmlist):
    x0, y0 = lmlist[0][1], lmlist[0][2]
    x1, y1 = lmlist[1][1], lmlist[1][2]
    x2, y2 = lmlist[2][1], lmlist[2][2]
    x3, y3 = lmlist[3][1], lmlist[3][2]
    x4, y4 = lmlist[4][1], lmlist[4][2]
    if y4 < y3 < y2 < y1 < y0:
        return True
    return False

detector = htm.HandDetector(detection_conf=0.7)
operation_mode = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    print("Operation Mode = ", operation_mode)
    if len(lmlist) != 0:
        print(lmlist)
        x4, y4 = lmlist[4][1], lmlist[4][2]
        x8, y8 = lmlist[8][1], lmlist[8][2]
        length = math.hypot(x8-x4, y8-y4)
        print(length)
    if operation_mode == 0:
        cv2.putText(img, "Command Mode", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                         (255, 255, 0), 2)
        cv2.putText(img, "1) Mouse Controller Mode", (10, 130), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 239, 255), 2)
        cv2.putText(img, "2) Volume Controller Mode", (10, 160), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 239, 255), 2)
        cv2.putText(img, "3) PPT Controller Mode", (10, 190), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 239, 255), 2)
        cv2.putText(img, "4) Media Player Mode", (10, 220), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 239, 255), 2)
        cv2.putText(img, "5) System Mode", (10, 250), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 239, 255), 2)
        if len(lmlist) != 0:
            x8, y8 = lmlist[8][1], lmlist[8][2]
            x4, y4 = lmlist[4][1], lmlist[4][2]
            length = math.hypot(x8-x4, y8-y4)
            print("Distance between 4 and 8 = ", length)
            cv2.putText(img, str(length), (100, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                         (255, 0, 255), 3)
            if one(lmlist) and (not two(lmlist)) and (not three(lmlist)) and (not four(lmlist)):
                # and (not two(lmlist)) and (not three(lmlist)) and (not four(lmlist)) and (not five(lmlist))
                x4, y4 = lmlist[4][1], lmlist[4][2]
                x8, y8 = lmlist[8][1], lmlist[8][2]
                length = math.hypot(x8-x4, y8-y4)
                cv2.putText(img, "One", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
                
                operation_mode = 1
            elif one(lmlist) and (lmlist) and (not three(lmlist)) and (not four(lmlist)):
                x4, y4 = lmlist[4][1], lmlist[4][2]
                x8, y8 = lmlist[8][1], lmlist[8][2]
                length = math.hypot(x8-x4, y8-y4)
                cv2.putText(img, "Two", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
                operation_mode = 2
            elif one(lmlist) and (lmlist) and three(lmlist) and (not four(lmlist)):
                x4, y4 = lmlist[4][1], lmlist[4][2]
                x8, y8 = lmlist[8][1], lmlist[8][2]
                length = math.hypot(x8-x4, y8-y4)
                cv2.putText(img, "Three", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
                operation_mode = 3
            elif (not one(lmlist)) and two(lmlist) and three(lmlist) and (not four(lmlist)):
                x4, y4 = lmlist[4][1], lmlist[4][2]
                x8, y8 = lmlist[8][1], lmlist[8][2]
                length = math.hypot(x8-x4, y8-y4)
                cv2.putText(img, "Four", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
                operation_mode = 4
            elif one(lmlist) and (lmlist) and three(lmlist) and four(lmlist) and five(lmlist):
                x4, y4 = lmlist[4][1], lmlist[4][2]
                x8, y8 = lmlist[8][1], lmlist[8][2]
                length = math.hypot(x8-x4, y8-y4)
                cv2.putText(img, "Five", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
                operation_mode = 5
    elif operation_mode == 1:
        if len(lmlist) != 0:
            x20, y20 = lmlist[20][1], lmlist[20][2]
            x4, y4 = lmlist[4][1], lmlist[4][2]
            length = math.hypot(x20 - x4, y20 - y4)
            print("Distance between 4 and 20 = ", length)
            if length > 200:
                operation_mode = 0

            x1, y1 = lmlist[8][1], lmlist[8][2]
            x2, y2 = lmlist[4][1], lmlist[4][2]
            left_click_button_length = math.hypot(x1 - x2, y1 - y2)
            if left_click_button_length < 20:
                pyautogui.click()
                pyautogui.sleep(1)
            # # Get the screen size
            # screen_width, screen_height = pyautogui.size()

            # # Calculate mouse position relative to screen size
            # mouse_x_cam = int((1 - x1) * wCam)
            # mouse_y_cam = int((1 - y1) * hCam)

            # mouse_x = int(mouse_x_cam * (screen_width / wCam))
            # mouse_y = int(mouse_y_cam * (screen_height / hCam))

            # margin = 10  # Adjust as needed
            # mouse_x = min(max(mouse_x, margin), screen_width - margin)
            # mouse_y = min(max(mouse_y, margin), screen_height - margin)

            # try:
            #     pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
            # except pyautogui.FailSafeException:
            #     print("Fail-safe triggered. Mouse moved to a corner of the screen.")
        frame = img
        # frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))

                if id == 1:
                    screen_x = screen_width * (1 - landmark.x)
                    screen_y = screen_height * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))


    elif operation_mode == 2:
        if len(lmlist) != 0:
            x20, y20 = lmlist[20][1], lmlist[20][2]
            x4, y4 = lmlist[4][1], lmlist[4][2]
            length = math.hypot(x20-x4, y20-y4)
            print("Distance between 4 and 20 = ", length)
            if length > 200:
                operation_mode = 0
            # print(lmlist[4], lmlist[8])

            x1, y1 = lmlist[4][1], lmlist[4][2]

            x2, y2 = lmlist[8][1], lmlist[8][2]

            cx, cy = (x1+x2)//2, (y1+y2)//2

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2-x1, y2-y1)
            # print(length)


            # Hand Range 50 - 130
            # Volume Range -63.5, 0.0

            vol = np.interp(length, [50, 125], [minVol, maxVol])
            volBar = np.interp(length, [50, 125], [400, 150])
            volPer = np.interp(length, [50, 125], [0, 100])
            print("Volume  = ", vol)
            volume.SetMasterVolumeLevel(vol, None)

            print(length, vol)
            if length<40:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (50, 150), (85, 400), (0,255,0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Volume Percentage: "+str(int(volPer))+"%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 255), 3)
    elif operation_mode == 3:
        if len(lmlist) != 0:
            x20, y20 = lmlist[20][1], lmlist[20][2]
            x4, y4 = lmlist[4][1], lmlist[4][2]
            length = math.hypot(x20-x4, y20-y4)
            print("Distance between 4 and 20 = ", length)
            if length > 200:
                operation_mode = 0
            if(one(lmlist) and (not two(lmlist)) and (not three(lmlist)) and (not four(lmlist))):
                pyautogui.hotkey('f5')
                print("f5")
                pyautogui.sleep(3)
            elif one(lmlist) and (lmlist) and (not three(lmlist)) and (not four(lmlist)):
                pyautogui.hotkey('right')
                print("Right")
                pyautogui.sleep(1)
            elif one(lmlist) and (lmlist) and three(lmlist) and (not four(lmlist)):
                pyautogui.hotkey('left')
                print("Left")
                pyautogui.sleep(1)
            

    elif operation_mode == 4:
        if len(lmlist) != 0:
            x20, y20 = lmlist[20][1], lmlist[20][2]
            x4, y4 = lmlist[4][1], lmlist[4][2]
            length = math.hypot(x20-x4, y20-y4)
            print("Distance between 4 and 20 = ", length)
            if length > 200:
                operation_mode = 0
            print("Distance between 4 and 20 = ", length)
            if length > 200:
                operation_mode = 0
            if(one(lmlist) and (not two(lmlist)) and (not three(lmlist)) and (not four(lmlist))):
                pyautogui.hotkey('space')
                print("space")
                pyautogui.sleep(1)
            elif one(lmlist) and (lmlist) and (not three(lmlist)) and (not four(lmlist)):
                pyautogui.hotkey('right')
                print("Right")
                pyautogui.sleep(1)
            elif one(lmlist) and (lmlist) and three(lmlist) and (not four(lmlist)):
                pyautogui.hotkey('left')
                print("Left")
                pyautogui.sleep(1)
            elif (not one(lmlist)) and two(lmlist) and three(lmlist) and (not four(lmlist)):
                pyautogui.hotkey('up')
                print("Up")
                pyautogui.sleep(1)
            elif one(lmlist) and (lmlist) and three(lmlist) and four(lmlist) and five(lmlist):
                pyautogui.hotkey('down')
                print("Down")
                pyautogui.sleep(1)
    elif operation_mode == 5:
        if len(lmlist) != 0:
            x20, y20 = lmlist[20][1], lmlist[20][2]
            x4, y4 = lmlist[4][1], lmlist[4][2]
            length = math.hypot(x20-x4, y20-y4)
            print("Distance between 4 and 20 = ", length)
            if length > 200:
                operation_mode = 0

            if (one(lmlist) and (not two(lmlist)) and (not three(lmlist)) and (not four(lmlist))):
                cv2.putText(img, "Screenshot Taken", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
                print("Screenshot")
                f = open("screenshot_number.txt", "r+")
                x = int(f.readline())
                print("Screenshot Number =",x)
                print(type(x))
                f.close()
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), 
                        cv2.COLOR_RGB2BGR)
                f = open("screenshot_number.txt", "w+")
                x+=1
                f.write(str(x))
                print("Screenshot Number After Iteration =",x)
                print(type(x))
                f.close()
                cv2.imwrite("./Screenshots/image"+str(x)+".png", image)
                c+=1
                time.sleep(3)
            else:
                print("")
                
    # if len(lmlist) != 0:
    #     x8, y8 = lmlist[8][1], lmlist[8][2]
    #     x4, y4 = lmlist[4][1], lmlist[4][2]
    #     x20, y20 = lmlist[20][1], lmlist[20][2]
    #     length = math.hypot(x20-x4, y20-y4)
    #     cv2.putText(img, str(length), (100, 100), cv2.FONT_HERSHEY_PLAIN, 3,
    #                     (255, 0, 255), 3)
    #     if thumbs_down(lmlist) and length < 92:
    #         cv2.putText(img, "Thumbs Down", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #                     (255, 0, 255), 3)
    #     elif thumbs_up(lmlist) and length < 92:
    #         cv2.putText(img, "Thumbs Up", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #                     (255, 0, 255), 3)
    #     elif one(lmlist) and (not two(lmlist)) and(not three(lmlist)):
    #         x4, y4 = lmlist[4][1], lmlist[4][2]
    #         x8, y8 = lmlist[8][1], lmlist[8][2]
    #         length = math.hypot(x8-x4, y8-y4)
    #         cv2.putText(img, "One", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #                     (255, 0, 255), 3)
    #     elif one(lmlist) and (lmlist) and(not three(lmlist)):
    #         x4, y4 = lmlist[4][1], lmlist[4][2]
    #         x8, y8 = lmlist[8][1], lmlist[8][2]
    #         length = math.hypot(x8-x4, y8-y4)
    #         cv2.putText(img, "Two", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #                     (255, 0, 255), 3)
    #     elif one(lmlist) and (lmlist) and three(lmlist):
    #         x4, y4 = lmlist[4][1], lmlist[4][2]
    #         x8, y8 = lmlist[8][1], lmlist[8][2]
    #         length = math.hypot(x8-x4, y8-y4)
    #         cv2.putText(img, "Three", (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #                     (255, 0, 255), 3)
            
    # if len(lmlist)!=0:
    #     # print(lmlist[4], lmlist[8])
    #     print(lmlist)
    #     x1, y1 = lmlist[8][1], lmlist[8][2]

    #     x2, y2 = lmlist[20][1], lmlist[20][2]

    #     cx, cy = (x1+x2)//2, (y1+y2)//2

    #     # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    #     # cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
    #     # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    #     # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    #     length = math.hypot(x2-x1, y2-y1)
    #     print(length)
    #     if length<40:
    #         cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    #         print("Not Screenshot")
    #     else:
    #         print("Screenshot")
    #         image = pyautogui.screenshot()
    #         image = cv2.cvtColor(np.array(image), 
    #                  cv2.COLOR_RGB2BGR)
    #         # cv2.imwrite("C:/Users/anubr/Python Projects/Machine Vision/Project/Screenshots/image"+str(c)+".png", image)
    #         c+=1
    #         # time.sleep(3)

    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
    cv2.imshow("Img",img)
    cv2.waitKey(1)