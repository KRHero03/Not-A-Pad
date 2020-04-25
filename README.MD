# Not-A-Pad

Human Interaction with computers have evolved from time to time. In the era when computer was a device known to just a few, we interacted with computers with inserting and removing cables. 

Thereafter, we invented the mechanism of key pressed based devices, commonly known as keyboards, after which, came the pointing devices known as mouse.

This project aims to take the idea of pointing devices to a step further where the user can use their finger tips as a source of cursor tracking system.

It is my first project in the field of Computer Vision and Basic Image Processing which is also a part of ACM Month of Code 2020.

## Tools Used
 - python 3.7.3
 - opencv-python 4.2.0.34
 - numpy 1.18.2
 - pyautogui 0.9.50

## Installation

 - Clone the repository in your PC:-
    ```sh
    $ git clone https://github.com/KRHero03/Not-A-Pad.git
    ```
    
 - Install [Python][PythonURL] (Skip these steps and jump to "Run" if already done). 
 - Set Path to Python as environment variable in Windows machine.
 - Install required Python Modules by running the following commands in Terminal/Shell:-
    ```sh
    $ pip install pyautogui
    $ pip install opencv-python
    $ pip install numpy
    ```
 - Run the file main.py:-
    ```sh
    $ python main.py
    ```
 - Read the Instructions in your CLI and use the tool accordingly.

## How It Works
Not-A-Pad is a CLI based Python software that is made in 3 phases:
 - Capturing and filtering Images through Webcam
 - Detecting finger tips in the captured Images
 - Using detected finger tips and their movements to move and perform basic mouse cursor functions

#### Step 1: Capturing and filtering Images through Webcam
 - OpenCV is used to capture images every 34 milliseconds. Every Image is then feed to a thresholder which is then approximated using a Gaussian Blur filter using 3 x 3 Kernal. 

 - I have tried many filtering methods such as creating Histogram of the pixels present samples present in the rectangle box and then based on the Histogram identifying skin coloured regions, but this has been slightly inaccurate.
 - The most successful method for me was to allow users to set the value of Hue and Saturation which allows them to clearly view their Hands in WHITE and the background in BLACK.
 - This is also known as Hand Calibration.

<p align="center"><image src="Images/1.png"></p>

#### Step 2: Detecting finger tips in the captured Images

 - After Hand Calibration, the filtered mask Image is then used to detect contours.
 
 - [Contours][ContourOpenCV] are surfaces with similar colour or intensity(In our case, WHITE). These contours are marked using points which form a closed polygon.
 - A Contour whose area is maximum is assumed as the hand in our case, namely, the MaxContour.
 - We also find the centroid of this MaxContour using moment of Intensities of all pixels. More information can be found here.
 - A convex hull is created for this MaxContour using a fusion of Jarvis and Graham Scan algorithm whose implementation details can be found [here][ConvexHull]. 
 - This convex hull returns us a list of points which consists of start, end and far points. More details can be found [here][ConvexHullOpenCV].
 - Now there were two choices for me at this stage:- Either to use ConvexityDefects in order to find the [defects][DefectsOpenCV] (far points in MaxContours from Convex Hull Boundary) in order to detect finger tips or loop through the convex hull points and detect acute angles using Cosine formula.
 - ~~I tried both and found better results in the second choice. For an interval of every 16 points(Used this number using trial and error) in MaxContour, I found the Cosine between a point P and P + 40. I also found the Cosine between  P - 16 & P + 40 and P + 16 & P +40 and found the maximum Cosine among the three.~~
 - ~~Also, I found the rotation between P & P + 40.~~
 - ~~Now if the rotation is only on one side and the cosine is less than 0.5 (Acute Angles), then we assume it as a finger.~~
 - I tried another method using defects. In this method, I find the defects and find the cosine of angle between far,start and end. If it is greater than a certain amount and if depth 'd' is greater than a certain amount, then a finger is detected. Also, this method doesn't detect a single finger because for that I need at least one defect which cannot be sufficiently produced by a single finger. Instead of single finger detection, I added the farthest point detection.
 - This algorithm has a flaw because it detects the angles formed at the edges of the Image when the hand is intersecting with the edge. It can be solved using an offset from the edges, which in my case, was 20.
 - This concludes the finger tip detection phase. It was by far the most complex step in this project.

<p align="center"><image src="Images/2.png"></p>

As shown in the Image, Finger Tips are detected as Green circles connected to Centroid marked as Red Circle via Green Lines. The Convex Hull is detected as Red Closed Convex Polygon

#### Step 3: Using detected finger tips and their movements to move and perform basic mouse cursor functions

 - It is really simple after we are able to detect the Finger tips. This step utilises [PyAutoGui][Pyautogui] Library

 - If there is only one Finger Tip, we track its movement and supply it to the mouse cursor.
 - If there are two Finger Tips, we invoke a Double Click event.
 - If there are three Finger Tips, we invoke a Scroll event.


## Drawbacks

 - The tool requires User Input to counter light sensitive backgrounds. Without User Input for determining the values of Hue and Saturation thresholds, the performance in finger detection would have reduced. Also, the tool is based on Image processing and colour detection which may make finger detection a little difficult in backgrounds similar to skin tone.
 
 - The conversion of finger movements to cursor movements is very coarse because I was not able to find the exact equation that related the Finger Movements to Cursor. It is similar to how a trackpad is designed to work. It involves determination of finger acceleration and mapping it to pointer speed, which was currently beyond my scope of knowledge for now. 

I will make sure to improve upon it in due course of time.
 
## Suggestions

Any suggestion or advice will be appreciated. 
Feel free to contact me at krunalrank0609@gmail.com



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[PythonURL]: <https://www.python.org/>
[ConvexHull]: <https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/>
[ConvexHullOpenCV]: <https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html>
[DefectsOpenCV]: <https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html>
[MomentOpenCV]: <https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments>
[ContourOpenCV]: <https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html>
[Pyautogui]: <https://pypi.org/project/PyAutoGUI/>


