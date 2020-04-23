from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import pyautogui
import math


max_value = 255
max_value_H = 255
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Live'
window_detection_name = 'Hand Calibration'
low_H_name = 'Low H'
low_S_name = 'Low S'
high_H_name = 'High H'
high_S_name = 'High S'

offset = 20

mouseMode = False

        
screenSize = pyautogui.size()

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def showHandCalibration(frame):
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)
    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.imshow(window_detection_name, frame)   

def moveMouse(farthestPoint, frame):
        if mouseMode==True:
            targetX = farthestPoint[0] - offset - 10
            targetY = farthestPoint[1] - offset - 10 
            pyautogui.moveTo(targetX*screenSize[0]/(frame.shape[1]-offset -10), targetY*screenSize[1]/(frame.shape[0]-offset-10))

def doubleClick():
    if mouseMode==True:
        pyautogui.doubleClick()

def scroll():
    if mouseMode==True:
        pyautogui.scroll(10)

def filterForeground(frame):
    return
    # cv.fastNlMeansDenoising(frame,frame)
    # cv.GaussianBlur(frame,(3,3),0,frame)
    # cv.medianBlur(frame,3,frame)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    # frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # frame = cv.morphologyEx(frame, cv.MORPH_GRADIENT, kernel)

def findPointDistance(pt1,pt2):
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)   

def getMaxContour(contours):
    if len(contours) > 0:
        maxIndex = 0
        maxArea = 0

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv.contourArea(cnt)

            if area > maxArea:
                maxArea = area
                maxIndex = i
        return contours[maxIndex]

def getContours(histMask):
    ret, thresh = cv.threshold(histMask, 0, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    
    return contours

def getCentroid(contour):
        moment = cv.moments(contour)
        if moment['m00'] != 0:
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            return cx, cy
        else:
            return None

def angle(contour,i,r):
    length = len(contour)
    if length==0:
        return 0
    point0 = contour[i%length][0] if i>0 else contour[(length-1-r%length+length)%length][0]
    point1 = contour[(i+r)%length][0]
    point2 = contour[i-r][0] if i>r else contour[(length-1-r%length+length)%length][0]

    ux = point0[0] - point1[0]
    uy = point0[1] - point1[1]
    vx = point0[0] - point2[0]
    vy = point0[1] - point2[1]

    return (ux*vx + uy*vy)/(((ux*ux + uy*uy + 0.00000000001)*(vx*vx + vy*vy + 0.00000000001))**0.5)

def rotation(contour,i,r):
    length = len(contour)
    if length==0:
        return 0
    point0 = contour[i%length][0] if i>0 else contour[(length-1-r%length+length)%length][0]
    point1 = contour[(i+r)%length][0]
    point2 = contour[i-r][0] if i>r else contour[(length-1-r%length+length)%length][0]

    ux = point0[0] - point1[0]
    uy = point0[1] - point1[1]
    vx = point0[0] - point2[0]
    vy = point0[1] - point2[1]

    return (ux*vy - vx*uy)


cap = cv.VideoCapture(0,cv.CAP_DSHOW)

calibrated = False

print('Not-A-Pad')
print('Cursor movement and tracking system using Hand Gestures')
print('Made by Krunal Rank')
print('')

print('------------------------------------------Instructions-----------------------------------------------')
print('1. Calibrate your hand using the Hand Calibration window so that the software recognises your hand.')
print('   Please set the trackbar values so that your Hand, in front of your WebCam appears in WHITE and background appears in BLACK.')
print('   Suggested Values:-')
print('   Low H: 0','High H: 14','Low S: 45','High S: 255')
print('   You need not follow the above values. They are just for reference.')
print('2. Press "c" to confirm Hand Calibration')
print('3. Verify your Hand Calibration in "Live" window. Press "c" to recalibrate.')
print('4. Press "m" to enable or disable Mouse Movement. Note that your screen cursor will move accordingly.')
print('')

print('Controls:')
print('One Finger Detected - Uses its movement as Cursor Movement')
print('Two Fingers Detected - Double Click')
print('Three Fingers Detected - Scroll according to Cursor Position')
print('Esc - Exit to Calibration or Exit application')
print('c - Toggle Hand Calibration')
print('m - Toggle Mouse Movement')

while True:
    
    ret, frame_derived = cap.read()    
    rows,cols,channels = frame_derived.shape
    frame = frame_derived[:rows//2,:cols//2,:].copy()
    frame = cv.flip(frame,1)
    if frame is None:
        break



    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_GRAY = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame_GRAY = cv.bitwise_and(frame_GRAY,frame_threshold)
    cv.inRange(frame_GRAY,20,255,frame_GRAY)
    foreground = frame_GRAY.copy()
    if(calibrated==False):
        showHandCalibration(foreground)
    else:
        filterForeground(foreground)

        contours = getContours(foreground)
        maxContour = getMaxContour(contours)
        
        if(maxContour is not None):

            # cv.drawContours(frame,[maxContour],0,(255,0,0),1)

            hullPoints = cv.convexHull(maxContour,returnPoints=True)
            hullInts = cv.convexHull(maxContour,returnPoints=False)
            
            centroid = getCentroid(maxContour)

            fingerPoints = []
            step = 16
            r = 40
            i = 0
            while i< len(maxContour):
                cos0 = angle(maxContour,i,r)

                if cos0 >0.5 and step+i<len(maxContour):
                    cos1 = angle (maxContour, i - step, r)
                    cos2 = angle (maxContour, i + step, r)
                    maxCos = max(max(cos0, cos1), cos2)
                    temp = rotation (maxContour, i, r)
                    if maxCos==cos0 and temp<0: 
                        if(maxContour[i][0][0]>offset and maxContour[i][0][1]>offset and abs(maxContour[i][0][0]-foreground.shape[1])>offset and abs(maxContour[i][0][1]-foreground.shape[0])>offset ):
                            fingerPoints.append(maxContour[i][0])

                i = i + step
            

            # if(len(hullInts)>3):
            #     defects = cv.convexityDefects(maxContour,hullInts)
            
            # if defects is None:
            #     continue

            # for defect in defects:
            #     (start,end,far,d)= defect[0]
                
            #     cv.circle(frame,tuple(maxContour[end][0]),5,(0,0,255),-1)                
            #     cv.circle(frame,tuple(maxContour[far][0]),5,(0,255,255),-1)
            #     cv.circle(frame,tuple(maxContour[start][0]),5,(0,255,0),-1)
            if(len(fingerPoints)==2):
                doubleClick()
            elif(len(fingerPoints)==1):
                moveMouse(fingerPoints[0],foreground)
            elif(len(fingerPoints)==3):
                scroll()


            for point in fingerPoints:
                (x,y) = (point[0],point[1])
                cv.circle(frame,(x,y),20,(0,255,0),2)
                cv.line(frame,centroid,(x,y),(0,255,0),4)
            cv.circle(frame,centroid,20,(0,0,255),2)
            cv.drawContours(frame,[hullPoints],0,(0,0,255),2)
        
    
    cv.imshow(window_capture_name, frame)
    # cv.imshow('Mask', foreground)
    key = cv.waitKey(34)
    if key == ord('q') or key == 27:
        if(calibrated):
            calibrated=False
        else:            
            break
    elif key==ord('c'):
        if(calibrated):                  
            calibrated=False
        else:
            calibrated=True
            cv.destroyWindow(window_detection_name)
    elif key==ord('m'):
        mouseMode = False if mouseMode is True else True