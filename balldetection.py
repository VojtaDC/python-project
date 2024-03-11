import cv2
from matplotlib import pyplot as plt
import numpy as np

#PATH = 'ball_maze.jpeg'    
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    #img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap='gray')
    newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(newimg, cv2.HOUGH_GRADIENT, 1, 2000, param1=200, param2= 20, minRadius=40, maxRadius=80)
    if circles is not None:
        for x, y, r in circles[0]:
            cv2.circle(img, (int(x), int(y)), int(r), (0,0,0), 3)
            #c = plt.Circle((x, y), r, fill=False, lw=1, ec='C1')
            #plt.gca().add_patch(c)
    else:
        print("none")
    #plt.gcf().set_size_inches((12, 8))
    #plt.show()
    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
