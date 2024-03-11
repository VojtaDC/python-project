#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:14:49 2021

@author: bkeelson
"""
import numpy as np
import cv2 
# from BREADTH_FIRST_prototype import *
import BREADTH_FIRST_prototype as bf
#import testklik
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
import time

test_hue = None


# Functie die wordt aangeroepen bij muisklik
def get_position(event, x, y, flags, color):
    global test_hue
    if event == cv2.EVENT_LBUTTONDOWN:  # Als er op de linkermuisknop wordt geklikt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converteer het frame naar HSV
        color = hsv[y, x]  # Haal de kleur op van de pixel waarop is geklikt
        global test_hue
        test_hue = color[0]
        print(color)
        
def color_ranges(test_color):
    if test_hue > 165:
        lower_red = np.array([0, 50, 20]) 
        upper_red = np.array([test_hue-165, 255, 255])
        
        lower_red2 = np.array([test_hue-15, 50, 20]) 
        upper_red2 = np.array([180, 255, 255])
        
        return lower_red, upper_red, lower_red2, upper_red2
    
    # Range for upper range
    elif test_hue < 15:
        lower_red = np.array([0, 50, 20]) 
        upper_red = np.array([test_hue+15, 255, 255])
        
        lower_red2 = np.array([180-test_hue, 50, 20]) 
        upper_red2 = np.array([180, 255, 255])
        
        return lower_red, upper_red, lower_red2, upper_red2
    else:
        lower_red = np.array([test_hue-15, 50, 20]) 
        upper_red = np.array([test_hue+15, 255, 255])
        
        return lower_red, upper_red, None, None
    
def find_closest_skeleton_point_with_kdtree(path, skeleton):
    # Get the coordinates of all skeleton points
    skeleton_points = np.argwhere(skeleton == 1)

    # Create a KDTree
    tree = KDTree(skeleton_points)

    closest_points = []
    for path_point in path:
        # Query the KDTree to find the closest skeleton point
        distance, index = tree.query(path_point)
        closest_skeleton_point = tuple(skeleton_points[index])
        closest_points.append(closest_skeleton_point)

    return closest_points
    
    
        

if __name__ == "__main__":
    # Maak een nieuw venster
    cv2.namedWindow("Video Feed")
    
    # Stel de muiscallback functie in op get_position
    cv2.setMouseCallback("Video Feed", get_position)
    #######
    
    # setup webcam feed 
    # cap = cv2.VideoCapture(0)  # Change this line to capture video from webcam
    
    kernel = np.ones((5,5), np.uint8)
    
    # # frame = cv2.imread('/Users/vojtadeconinck/Downloads/python-project/Labyrinth.jpeg')
    # ret, foto = cap.read()
    foto = cv2.imread('/Users/vojtadeconinck/Downloads/python-project/TestNieuwekleuren.jpg')
    frame = cv2.GaussianBlur(foto, (5,5), 0)
    
    while test_hue is None:
        
        # _, frame = cap.read()
        
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", frame)
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
         
    x_ranges = color_ranges(test_hue)
    
    crop = None
    ## loop to continuously acquire frames from the webcam
    
    
       # _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    red_mask = cv2.inRange(hsv_frame, x_ranges[0], x_ranges[1])
    if x_ranges[2] is not None:
        mask2 = cv2.inRange(hsv_frame, x_ranges[2], x_ranges[3])
        red_mask += mask2
    
    # Generating the final mask to detect red color
    red_mask = cv2.erode(red_mask, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=10)
    red_mask = cv2.erode(red_mask, kernel, iterations=5)

    
    
    # Assuming red_mask is your image
    coords = cv2.findNonZero(red_mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image to the found coordinates
    crop = red_mask[y+30:y+h-30, x+30:x+w-30]
    crop = cv2.resize(crop, None, fx = 0.5, fy = 0.5)
    anticrop = np.logical_not(crop)
    # Skeletonize the image
    skeleton = skeletonize(anticrop)
    skeleton = (skeleton.astype(np.uint8))
    inverse_skeleton = np.logical_not(skeleton)
    
    second_kernel = np.ones((7,7), np.uint8)
    skeleton = cv2.dilate(skeleton, second_kernel, iterations=1)
    
    
    # cv2.imshow("Video Feed", inverse_skeleton)
    
    # cv2.waitKey(300000000)
    start = (len(crop)//2,0)
    print("start zonder skel= ", start)
    end = (len(crop)//2, round(len(crop[0])*(174/179)))
    print("end zonder skel = ", end)
   
    
    start = find_closest_skeleton_point_with_kdtree([start], skeleton)[0]
    print("start = ", start)
    end = find_closest_skeleton_point_with_kdtree([end], skeleton)[0]
    
    print("end = ", end, inverse_skeleton[end[0]][end[1]], skeleton[end[0]][end[1]])
    start_time = time.time()
    distances = bf.breadth_first(inverse_skeleton, start, end)
    for i in range(len(distances)):
        for a in range(len(distances[i])):
            if distances[i][a] != -1:
                print(distances[i][a])

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    print(len(distances))
    path = bf.print_shortest_path(distances, start, end)
    # path = find_closest_skeleton_point_with_kdtree(path, skeleton)
    print('jep')
    color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    print(len(path))
    for i in range(len(path) - 1):
        x , y = path[i]
        point1 = (int(y), int(x))
        x , y = path[i + 1]
        point2 = (int(y), int(x))
        cv2.line(color_frame, point1, point2, (0, 0, 255), 2)
        cv2.circle(color_frame, point1, 4, (0,255,0), 2)
    
    cv2.imshow("Video Feed", color_frame)
    
    
    while False:
     
        #cv2.imshow("Frame", frame)
        
        # cv2.imshow("Video Feed", )
        # start = crop
        
        cv2.imshow("Video Feed", crop)
        
        
         
        
    
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print('nu wachten we')
    cv2.waitKey(100000)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    #Voor centreren: centerlines --> Lloris zegt: vindt een vorm en pakt dan het midden van de vorm git config --global --edit
