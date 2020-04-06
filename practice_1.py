import cv2
import numpy as np
import pyautogui



np.set_printoptions(threshold=np.Infinity)
pyautogui.FAILSAFE = False
class Detector:
    calibrated = False
    hist = None
    faceClassifier = cv2.CascadeClassifier('frontal_face.xml')
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=30, detectShadows=False)
    mouseMode = True
    def __init__(self):
        # hLowThreshold = 0
        # hHighThreshold = 0
        # sLowThreshold = 0
        # sHighThreshold = 0
        # vLowThreshold = 0
        # vHighThreshold = 0
        
        self.mouseMode = True
        self.traversePoints = []
        
        screenSize = pyautogui.size()
        self.screenSizeX = screenSize[0]
        self.screenSizeY = screenSize[1]

        self.calibrated = False
        rectSize = 0
        rect1 = None
        hist = None

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

        hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        self.hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    ''' def calculateThreshold(self,sample1,sample2):
        offsetLowThreshold = 80
        offsetHighThreshold = 30

        mean1 = cv2.mean(sample1)
        mean2 = cv2.mean(sample2)

        # print(mean1[0],mean2[0])
        self.hLowThreshold = (mean1[0] if mean1[0]<mean2[0] else mean2[0])- offsetLowThreshold
        self.hHighThreshold = (mean1[0] if mean1[0]<mean2[0] else mean2[0]) + offsetHighThreshold

        # print(mean1[1],mean2[1])
        self.sLowThreshold = (mean1[1] if mean1[1]<mean2[1] else mean2[1])- offsetLowThreshold
        self.sHighThreshold = (mean1[1] if mean1[1]<mean2[1] else mean2[1]) + offsetHighThreshold

        self.vLowThreshold = 0
        self.vHighThreshold = 255 '''

    def getSkinMask(self,frame):
        if(self.calibrated==False):
            return np.zeros(frame.shape)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv_frame], [0, 1], self.hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        # thresh = cv2.dilate(thresh, kernel, iterations=5)
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

    def execute(self, farthestPoint, frame):
        if self.mouseMode:
            targetX = farthestPoint[0]
            targetY = farthestPoint[1]
            #pyautogui.moveTo(targetX,targetY)
            pyautogui.moveTo(targetX*self.screenSizeX/frame.shape[1], targetY*self.screenSizeY/frame.shape[0])
        # elif self.scrollMode:
        #     if len(self.traversePoints) >= 2:
        #         movedDistance = self.traversePoints[-1][1] - self.traversePoints[-2][1]
        #         pyautogui.scroll(-movedDistance/2)

    def drawPath(self, frame, traversePoints):
        for i in range(1, len(self.traversePoints)):
            thickness = int((i + 5)/5)
            cv2.line(frame, traversePoints[i-1], traversePoints[i], [255, 0, 0], thickness)
        
        

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

            self.drawPath(frame, self.traversePoints)
            self.execute(farthestPoint, frame)
            cv2.imshow('TrackPad',frame)


    def backgroundSubtractor(self,frame):    
        fgmask = self.bgSubtractor.apply(frame, learningRate=0) 
        kernel = np.ones((4, 4), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.bitwise_and(frame, frame, mask=fgmask)

    def removeFaces(self,frame):
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray_frame_equalised = cv2.equalizeHist(gray_frame)

            rects = self.faceClassifier.detectMultiScale(
            gray_frame_equalised, scaleFactor=1.1, minNeighbors=2, minSize=(120, 120),
            flags=0 | cv2.CASCADE_SCALE_IMAGE)

            for x, y, w, h in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -10)

capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = Detector()
while capture.isOpened():
    _,frame_derived = capture.read()
    rows,cols,channels = frame_derived.shape
    frame = frame_derived[:2*rows//5,:2*cols//5,:].copy()
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
        detector.calibrate(frame)
        print('Hand Calibrated!')
    elif(key==27):   
        if(detector.calibrated==True):
            print('Hand Calibration Removed!')
            detector.calibrated==False
        else:
            break

cv2.destroyAllWindows()