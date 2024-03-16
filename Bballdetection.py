import cv2
from matplotlib import pyplot as plt
import numpy as np

#PATH = 'ball_maze.jpeg'    
def Bballdetection(frame):
    coordinates = []
    newimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(newimg, cv2.HOUGH_GRADIENT, 1, 10000, param1=200, param2= 40, minRadius=2, maxRadius=4)
    if circles is not None:
        for x, y, r in circles[0]:
            coordinates.append([int(x),int(y),int(r)])
            
        # print('aaaaaaaaa')
        return coordinates
    
    else:
        print("none")
        return None
#checkcheck [(55, 651), (54, 576), (94, 576), (96, 585), (96, 617), (98, 619), (98, 650), (100, 656), (232, 655)]