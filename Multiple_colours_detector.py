#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2 
import BREADTH_FIRST_prototype as bf



test_hue = None


# Functie die wordt aangeroepen bij muisklik
def get_position(event, x, y, flags, color):
    global test_hue
    if event == cv2.EVENT_LBUTTONDOWN:  # Als er op de linkermuisknop wordt geklikt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converteer het frame naar HSV
        color = hsv[y, x]  # Haal de kleur op van de pixel waarop is geklikt
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
    
    
        

if __name__ == "__main__":
    # Maak een nieuw venster
    cv2.namedWindow("Video Feed")
    
    # Stel de muiscallback functie in op get_position
    cv2.setMouseCallback("Video Feed", get_position)
    #######
    
    # setup webcam feed 
    cap = cv2.VideoCapture(0)  # Change this line to capture video from webcam
    
    kernel = np.ones((5,5), np.uint8)
    
    # frame = cv2.imread('/Users/vojtadeconinck/Downloads/python-project/Labyrinth.jpeg')
    ret, foto = cap.read()
    frame = cv2.GaussianBlur(foto, (5,5), 0)
    
    while test_hue is None:
        
        _, frame = cap.read()
        
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
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)
    
    
    # Assuming red_mask is your image
    coords = cv2.findNonZero(red_mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image to the found coordinates
    crop = red_mask[y:y+h, x:x+w]
    crop = cv2.resize(crop, None, fx = 0.5, fy = 0.5)
    
    cv2.imshow("Video Feed", crop)
    
    cv2.waitKey(1)
    
    distances = bf.breadth_first(crop, (len(crop)//2,0), (len(crop)//2,len(crop[0])) )
    path = bf.print_shortest_path(distances, (len(crop)//2,0), (len(crop)//2, round(len(crop[0])*(174/179)) ))
    
    color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    for element in path:
        x,y = element
        cv2.circle(color_frame, (y,x), 2,(0,0,255))
    
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
