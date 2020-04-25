import cv2
import numpy as np

np.set_printoptions(threshold=np.Infinity)




class BackgroundRemover:
    calibrated = False
    def __init__(self):
        background = None

    def calibrate(self,frame):
        self.background = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Background',self.background)
        self.calibrated = True

    def getForeground(self,frame):
        foregroundMask = self.getForegroundMask(frame)
        foreground = frame.copy()
        foreground[:,:,0] = np.where(foregroundMask==0,frame[:,:,0],0)
        foreground[:,:,1] = np.where(foregroundMask==0,frame[:,:,1],0)
        foreground[:,:,2] = np.where(foregroundMask==0,frame[:,:,2],0)

        return foreground


    def getForegroundMask(self,frame):

        if self.calibrated is False:
            foregroundMask = np.zeros((frame.shape[0],frame.shape[1]))
            return foregroundMask
        
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        foregroundMask = self.removeBackground(gray_frame)

        return foregroundMask


    def removeBackground(self,foregroundMask):
        
        thresholdOffset = 13
        
        mask = cv2.inRange(foregroundMask,self.background-thresholdOffset,self.background+thresholdOffset)
        
        return mask
    

class FaceDetector:
    faceClassifier = None
    def __init__(self):
        self.faceClassifier = cv2.CascadeClassifier('frontal_face.xml')

    def removeFaces(self,frame):
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_frame_equalised = cv2.equalizeHist(gray_frame)

        rects = self.faceClassifier.detectMultiScale(
        gray_frame_equalised, scaleFactor=1.1, minNeighbors=2, minSize=(120, 120),
        flags=0 | cv2.CASCADE_SCALE_IMAGE)

        for x, y, w, h in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
    

capture = cv2.VideoCapture(0)

skinDetector = SkinDetector()
backgroundRemover = BackgroundRemover()
faceDetector = FaceDetector()

while capture.isOpened():
    _,frame = capture.read()
    frame_copy = frame.copy()
    skinDetector.draw_rect(frame_copy)
    
    foreground = backgroundRemover.getForeground(frame)  
    faceDetector.removeFaces(foreground)
    #faceDetector.removeFaces(frame,foreground)
    # handMask = skinDetector.getSkinMask(frame)

    cv2.imshow('Foreground',foreground)
    #cv2.imshow('HandMask',handMask)
    #faceDetector.removeFaces(frame)


    cv2.imshow('Live',frame_copy)

    
    
    key = cv2.waitKey(1)

    if (key==ord('c')):
        backgroundRemover.calibrate(frame)
    elif (key==ord('h')):
        skinDetector.calibrate(frame)
    elif (key==27):
        break

cv2.destroyAllWindows()


# Video Save _4_
# fourcc = cv2.VideoWriter_fourcc(*'XVID') 
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 

# Gaussian Blur Technique _1_
# frame_copy=cv2.GaussianBlur(frame,(3,3),50)
# cv2.imshow('Modified',frame_copy)

# Edge Detector _2_
# edge_1 = cv2.Canny(frame,10,100,None,3)
# cv2.imshow('Edges1',edge_1)
# edge_2 = cv2.Canny(frame,1,10,None,3)
# cv2.imshow('Edges2',edge_2)
# edge_3 = cv2.Canny(frame,10,500,None,3)
# cv2.imshow('Edges3',edge_3)
# edge_4 = cv2.Canny(frame,10,100,None,7)
# cv2.imshow('Edges4',edge_4)

# Gaussian Blur + Edge Detector _3_
# frame_copy = cv2.GaussianBlur(frame,(9,9),0)
# edge = cv2.Canny(frame_copy,10,100,3)
# cv2.imshow('Edges',edge)

# Video Save _4_
# out.write(frame)

# Frame Brightness _5_
# frame = frame + 15
# frame = 255 if frame.any()>255 else frame
# cv2.imshow('Modified',frame)

# Frame Coversion
# hsv_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
# gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# cv2.imshow('HSV',hsv_frame)
# cv2.imshow('GRAY',gray_frame)