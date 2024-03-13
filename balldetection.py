import cv2
from matplotlib import pyplot as plt
import numpy as np

#PATH = 'ball_maze.jpeg'    
def balldetection(frame):
    newimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(newimg, cv2.HOUGH_GRADIENT, 1, 2000, param1=200, param2= 20, minRadius=40, maxRadius=80)
    if circles is not None:
        for x, y, r in circles[0]:
            cv2.circle(frame, (int(x), int(y)), int(r), (0,0,0), 3)
        return circles[0]
    else:
        print("none")
        return None
