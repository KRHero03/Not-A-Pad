import cv2
import numpy as np
import pyautogui
import math
import sys


np.set_printoptions(threshold=np.Infinity)
pyautogui.FAILSAFE = False

class Detector:



    def __init__(self):
        self.hLowThreshold = 0
        self.hHighThreshold = 0
        self.sLowThreshold = 0
        self.sHighThreshold = 0
        self.vLowThreshold = 0
        self.vHighThreshold = 0
        
        self.calibrated = False

        rectSize = 0
        rect1 = None
        self.hist = None

        self.faceClassifier = cv2.CascadeClassifier('frontal_face.xml')
        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=20, detectShadows=False)

        self.fingerScaleLimit = 0.6
        self.neighbourScaleLimit = 0.05

        self.mouseMode = True

        self.traversePoints = []
        # self.filterStartPoints = []
        # self.filterFarPoints = []
        # self.filterFingerPoints = []

        screenSize = pyautogui.size()
        self.screenSizeX = screenSize[0]
        self.screenSizeY = screenSize[1]

    def draw_rect(self,frame):
        self.rectSize = 20

        (rows,cols,channels) = frame.shape

        self.rect1 = (cols/5,rows/2)     

        x1 = self.rect1[0]
        y1 = self.rect1[1]
        x11 = x1 + self.rectSize
        y11 = y1 + self.rectSize


        cv2.rectangle(frame,(int(x1),int(y1)),(int(x11),int(y11)), (0, 255, 0), 1)
    
    def calibrate(self,frame):
        
        self.calibrated = True
        
        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        roi = np.zeros([self.rectSize,self.rectSize,3])
        roi = hsv_frame[int(self.rect1[0]):int(self.rect1[0])+int(self.rectSize),int(self.rect1[1]):int(self.rect1[1])+int(self.rectSize),:]
        self.calculateThreshold(roi)

        hist = cv2.calcHist([roi], [0, 1], None, [100, 256], [0, 100, 0, 256])
        self.hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    def calculateThreshold(self,sample1):
        offsetLowThreshold = 80
        offsetHighThreshold = 30

        mean1 = cv2.mean(sample1)

        # print(mean1[0],mean2[0])
        self.hLowThreshold = mean1[0] - offsetLowThreshold
        self.hHighThreshold = mean1[0] + offsetHighThreshold

        # print(mean1[1],mean2[1])
        self.sLowThreshold = mean1[1] - offsetLowThreshold
        self.sHighThreshold = mean1[1] + offsetHighThreshold

        self.vLowThreshold = 0
        self.vHighThreshold = 255 

    def removeFaces(self,frame):
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_frame_equalised = cv2.equalizeHist(gray_frame)

        rects = self.faceClassifier.detectMultiScale(
        gray_frame_equalised, scaleFactor=1.1, minNeighbors=2, minSize=(120, 120),
        flags=0 | cv2.CASCADE_SCALE_IMAGE)

        for x, y, w, h in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -10)
        
        return frame

    def backgroundSubtractor(self,frame):    
        if(self.calibrate==False):
            return np.zeros(frame.shape)
        foreground = np.zeros(frame.shape)

        frame = cv2.GaussianBlur(frame,(11,11),0)
        cv2.imshow('BGSFrame1',frame)

        frame = cv2.medianBlur(frame,11)        
        cv2.imshow('BGSFrame2',frame)

        cv2.cvtColor(frame,cv2.COLOR_BGR2HSV,frame)        
        cv2.imshow('BGSFrame3',frame)
	    
        cv2.inRange(frame,(self.hLowThreshold,self.sLowThreshold,self.vLowThreshold),(self.hHighThreshold,self.sHighThreshold,self.vHighThreshold),foreground)
        cv2.imshow('BGSFrame',foreground)

        return foreground
        # fgmask = self.bgSubtractor.apply(frame, learningRate=0) 
        # kernel = np.ones((3,3), np.uint8)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=3)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)        
        # cv2.imshow('Background Removed',fgmask)
        # return cv2.bitwise_and(frame, frame, mask=fgmask)  

    def getSkinMask(self,frame):
        if(self.calibrated==False):
            return np.zeros(frame.shape)
        frame = cv2.GaussianBlur(frame,(5,5),0)
        frame = cv2.medianBlur(frame,5,0)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv_frame], [0, 1], self.hist, [0, 100, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=5)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

        thresh = cv2.merge((thresh, thresh, thresh))
        return cv2.bitwise_and(frame, thresh)

    def getMaxContours(self, contours):
        if len(contours) > 0:
            maxIndex = 0
            maxArea = 0

            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)

                if area > maxArea:
                    maxArea = area
                    maxIndex = i
            return contours[maxIndex]

    def getContours(self, histMask):
        grayHistMask = cv2.cvtColor(histMask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grayHistMask, 0, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def getCentroid(self, contour):
        moment = cv2.moments(contour)
        if moment['m00'] != 0:
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            return cx, cy
        else:
            return None

    def moveMouse(self, farthestPoint, frame):
        if self.mouseMode:
            targetX = farthestPoint[0]
            targetY = farthestPoint[1]
            #pyautogui.moveTo(targetX,targetY)
            if(len(self.traversePoints)>1):
                alpha = 0.5
                beta = (500,500)
                speed = [0,0]
                speed[0]= farthestPoint[0] - self.traversePoints[-2][0]
                speed[1] = farthestPoint[1] - self.traversePoints[-2][1]

                # modVector = (reqVectorX**2 + reqVectorY**2+ 0.00000005)**0.5

                # unitVectorX = math.exp(reqVectorX/modVector -1)
                # unitVectorY = math.exp(reqVectorY/modVector -1)
                
                #print(unitVectorX,unitVectorY)

                curPos = pyautogui.position()
                locationX = max(min(curPos[0] +speed[0],self.screenSizeX),0)
                locationY = max(min(curPos[1] + speed[1],self.screenSizeY),0)

                
                print(locationX,locationY)

                pyautogui.moveTo(locationX, locationY,0.010,pyautogui.easeInOutExpo)
            else:
                pyautogui.moveTo(targetX*self.screenSizeX/frame.shape[1], targetY*self.screenSizeY/frame.shape[0],0,pyautogui.easeOutQuad)

    def path(self, frame, traversePoints):
        for i in range(1, len(self.traversePoints)):
            thickness = int((i + 5)/5)
            cv2.line(frame, traversePoints[i-1], traversePoints[i], [255, 0, 0], thickness)
        if(len(self.traversePoints)>2000):
            self.traversePoints = [traversePoints[-2],traversePoints[-1]]
        
    def detect(self, frame):
        frame_copy = frame.copy()
        #self.removeFaces(frame_copy)
        #foreground = self.backgroundSubtractor(frame_copy)
        histMask = self.getSkinMask(frame_copy)
        contours = self.getContours(histMask)
        maxContour = self.getMaxContours(contours)

        centroid = self.getCentroid(maxContour)
        cv2.circle(frame, centroid, 5, [255, 0, 0], -1)

        if maxContour is not None:
            convexHull = cv2.convexHull(maxContour, returnPoints=False)
            defects = cv2.convexityDefects(maxContour, convexHull)
            farthestPoint = maxContour[maxContour[:,:,1].argmin()][0]
            # print("Centroid: {}, Farthest point: {}".format(centroid, farthestPoint))
            if farthestPoint is not None:
                # Reduce noise in farthestPoint
                if len(self.traversePoints) > 0:
                    if abs(farthestPoint[0] - self.traversePoints[-1][0]) < 10:
                        farthestPoint[0] = self.traversePoints[-1][0]
                    if abs(farthestPoint[1] - self.traversePoints[-1][1]) < 10:
                        farthestPoint[1] = self.traversePoints[-1][1]
                farthestPoint = tuple(farthestPoint)
                # print(farthestPoint)

                cv2.circle(frame, farthestPoint, 5, [0, 0, 255], -1)

                if len(self.traversePoints) < 10:
                    self.traversePoints.append(farthestPoint)
                else:
                    self.traversePoints.pop(0)
                    self.traversePoints.append(farthestPoint)

            self.path(frame, self.traversePoints)
            self.moveMouse(farthestPoint, frame)
            cv2.imshow('TrackPad',frame)

    def detectAll(self,frame):
        frame_copy = frame.copy()        
        # face_removed=self.removeFaces(frame_copy)
        frame_copy_foreground = self.backgroundSubtractor(frame_copy)
        histMask = self.getSkinMask(frame_copy_foreground)
        cv2.imshow('HistMask',histMask)
        contours = self.getContours(histMask)
        maxContour = self.getMaxContours(contours)

        if maxContour is not None:
            hullPoints = cv2.convexHull(maxContour,returnPoints=True)
            hullInts = cv2.convexHull(maxContour,returnPoints=False)
            if(len(hullInts)>3):
                defects = cv2.convexityDefects(maxContour,hullInts)
            else:
                return
            x,y,w,h = cv2.boundingRect(hullPoints)
            centerBoundingRectangle = self.getCentroid(maxContour)
            
            startPoints =[]
            farPoints = []

            if defects is None:
                return

            for defect in defects:
                defectInstance = defect[0]

                x1,y1 = maxContour[defectInstance[0]][0]
                startPoints.append((x1,y1))

                # x2,y2 = maxContour[defectInstance[1]][0]

                x3,y3 = maxContour[defectInstance[2]][0]

                if(self.findPointDistance((x3,y3),centerBoundingRectangle)<h*self.fingerScaleLimit):
                    farPoints.append((x3,y3))
                
                filteredStartPoints = self.approximatePosByMedian(startPoints,h*self.neighbourScaleLimit)
                filteredFarPoints = self.approximatePosByMedian(farPoints,h*self.neighbourScaleLimit)

                filteredFingerPoints = []

                # self.drawPointList(frame,filteredStartPoints,(0,0,255))
                # self.drawPointList(frame,filteredFarPoints,(0,255,0))

                if(len(filteredFarPoints)>1):
                    fingerPoints = []

                    for point in filteredStartPoints:

                        closestPoints = self.findClosestOnX(filteredFarPoints,point)

                        if(self.isFinger(closestPoints[0],point,closestPoints[1],5,60,centerBoundingRectangle,h*self.fingerScaleLimit)):
                            fingerPoints.append(point)
                    

                    if(len(fingerPoints)>0):
                        while(len(fingerPoints)>5):
                            fingerPoints.pop()
                        
                        for i in range(len(fingerPoints) -1 ):
                            if(self.findPointDistanceOnX(fingerPoints[i],fingerPoints[i+1])>h*self.neighbourScaleLimit):
                                filteredFingerPoints.append(fingerPoints[i])

                        if(len(fingerPoints)>2):

                            if(self.findPointDistanceOnX(fingerPoints[0],fingerPoints[len(fingerPoints)-1])>h*self.neighbourScaleLimit):
                                filteredFingerPoints.append(fingerPoints[len(fingerPoints)-1])
                        
                        else:
                            filteredFingerPoints.append(fingerPoints[len(fingerPoints)-1])            


                self.drawPointList(frame,farPoints,(0,255,255))
                cv2.circle(frame,centerBoundingRectangle,5,(255,0,0),-1)

        cv2.imshow('TrackPad',frame)
        
    

    def findPointDistance(self,pt1,pt2):
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

    def findPointDistanceOnX(self,pt1,pt2):
        return abs(pt1[0]-pt2[0])

    def drawPointList(self,frame,points,colorCode):
        i = int(0)
        for point in points:
            i = i + 1
            x,y = int(point[0]),int(point[1])
            cv2.circle(frame,(x,y),5,colorCode,-1)
            # cv2.putText(frame,str(i),(x,y),cv2.FONT_HERSHEY_PLAIN,3,colorCode,1)

    def findAngle(self,pt1,pt2,pt3):
        distance1 = self.findPointDistance(pt1,pt2)
        distance2 = self.findPointDistance(pt2,pt3)
        distance3 = self.findPointDistance(pt1,pt3)
        try:
            val = math.acos((distance1**2 + distance2**2 - distance3**2)/(2*distance1*distance2))*180/math.pi
        except:
            val = 0
        return val
    
    def approximatePosByMedian(self,points,maxDistance):
        medianPoints = []
        if(len(points)<=0 or maxDistance<=0):
            return medianPoints
        
        referencePoint = points[0]
        median = points[0]

        for point in points:
            if(self.findPointDistance(point,referencePoint)>maxDistance):
                medianPoints.append(median)
                referencePoint = point
                median = point
            else:
                median = ((point[0]+median[0])/2,(point[1]+median[1])/2)

        medianPoints.append(median)

        return medianPoints

    def findClosestOnX(self,points,pivot):
        finalPoints = []

        if(len(points)<=0):
            return finalPoints
        
        distanceX1 = sys.float_info.max
        distance1 = sys.float_info.max
        distance2 = sys.float_info.max
        point1 = points[0]
        for point in points:
            distanceX = self.findPointDistanceOnX(point,pivot)
            distance = self.findPointDistance(point,pivot)

            if(distance<distance1 and distanceX!=0 and distanceX<distanceX1):
                distanceX1 = distanceX
                distance1 = distance
                point1 = point
        

        finalPoints.append(point1)
        distanceXFirst =distanceX1

        distance1 = sys.float_info.max
        distanceX1 = sys.float_info.max
        point1 = points[0]

        for point in points:
            distanceX = self.findPointDistanceOnX(point,pivot)
            distance = self.findPointDistance(point,pivot)

            if(distance<distance1 and distanceX!=0 and distanceX<distanceX1 and distanceX != distanceXFirst):
                distanceX1 = distanceX
                distance1 = distance
                point1 = point

        finalPoints.append(point1)

        return finalPoints

    def isFinger(self,pt1,pt2,pt3,limitAngleInf,limitAngleSup,center,minDistanceFromPalm):
        angle = self.findAngle(pt1,pt2,pt3)

        if(angle>limitAngleSup or angle<limitAngleInf):
            return False
        
        deltaY1 = pt2[1] - pt1[1]
        deltaY2 = pt2[1] - pt3[1]

        if(deltaY1>0 or deltaY2>0):
            return False
            
        deltaY3 = center[1] - pt1[1]
        deltaY4 = center[1] - pt3[1]

        if(deltaY3<0 or deltaY4<0):
            return False

        distanceFromPalm = self.findPointDistance(pt2,center)

        if(distanceFromPalm<minDistanceFromPalm):
            return False
        
        distanceFromPalmFar1 = self.findPointDistance(pt1,center)
        distanceFromPalmFar2 = self.findPointDistance(pt3,center)

        if(distanceFromPalmFar1 < minDistanceFromPalm/4 or distanceFromPalmFar2 < minDistanceFromPalm/4):
            return False

        return True
    

capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = Detector()
while capture.isOpened():
    _,frame_derived = capture.read()
    rows,cols,channels = frame_derived.shape
    frame = frame_derived[:rows//2,:cols//2,:].copy()
    # frame = frame_derived.copy()
    frame_flipped = cv2.flip(frame,1)
    detector.draw_rect(frame)

    # frame_copy = frame_flipped.copy()
    # detector.removeFaces(frame_copy)
    # foreground = detector.backgroundSubtractor(frame_copy)
    # skinMask = detector.getSkinMask(foreground)

    if(detector.calibrated==True):
        detector.detect(frame_flipped)
    else:        
        cv2.imshow('TrackPad',frame)
    # cv2.imshow('Primary',frame_derived)
    #cv2.imshow('Flipped Frame',frame_flipped)

    key = cv2.waitKey(17)

    if(key==ord('c')):
        pass
    if(key==ord('m')):
        detector.mouseMode = True if detector.mouseMode==False else False
        if detector.mouseMode==True:            
            print('Mouse Mode enabled!')
        else:   
            print('Mouse Mode disabled!')
    elif(key==ord('h')):      
        if(detector.calibrated==True):
            print('Hand already calibrated! Press Esc to remove calibration and perform a new one!')  
        else:
            detector.calibrate(frame)
            print('Hand Calibrated!')
    elif(key==27):   
        if(detector.calibrated==True):
            print('Hand Calibration Removed!')
            detector.calibrated=False
        else:
            break

cv2.destroyAllWindows()