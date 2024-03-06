# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 08:47:53 2019

@author: bkeelson
"""

import cv2
import numpy as np

events= [i for i in dir(cv2) if 'EVENT' in i]
print (events)

drawing= False #when true mouse is pressed
mode= True # when true we draw rectangle else we draw a circle toggle using m
ix,iy=-1,-1
#mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,mode,drawing
    
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
        
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
                
            else:
                cv2.circle(img,(x,y),5,(255,0,0),-1)
                
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)     
        else:
            cv2.circle(img,(x,y),5,(255,0,0),-1)
            
        
img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k= cv2.waitKey(1) &0xFF
    if k== ord('m'):
        mode= not mode
    elif k== 27:
        break
    
cv2.destroyAllWindows()
