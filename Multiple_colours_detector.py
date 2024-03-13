
"""
Created on Tue Feb  9 20:14:49 2021

@authors: Stefan en Vojta
"""
import time
import cv2 
import numpy as np
import serial
import BREADTH_FIRST_prototype as bf
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
import PID_controller as pid

test_hue = None
start = None
end = None

# Functie die wordt aangeroepen bij muisklik
def get_position(event, x, y, flags, color):
    global test_hue
    if event == cv2.EVENT_LBUTTONDOWN:  # Als er op de linkermuisknop wordt geklikt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converteer het frame naar HSV
        color = hsv[y, x]  # Haal de kleur op van de pixel waarop is geklikt
        global test_hue
        test_hue = color[0]
        # print(color)

def pos_start(event, x, y, flags, color):
    global start
    if event == cv2.EVENT_LBUTTONDOWN:  # Als er op de linkermuisknop wordt geklikt
        global start
        start = (y, x)


def pos_end(event, x, y, flags, color):
    global end
    if event == cv2.EVENT_LBUTTONDOWN:
        global end
        end = (y, x)

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
    
def find_closest_skeleton_point_with_kdtree(path, padskelet):
    # Neem alle punten van padskelet 
    skeleton_points = np.argwhere(padskelet == 255)

    # Maak een KDTree
    tree = KDTree(skeleton_points)

    closest_points = []
    for path_point in path:
        # Vind het dichtsbijzijnde padskelet punt
        distance, index = tree.query(path_point)
        closest_skeleton_point = tuple(skeleton_points[index])
        closest_points.append(closest_skeleton_point)

    return closest_points

def balldetection(frame):
    newimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(newimg, cv2.HOUGH_GRADIENT, 1, 2000, param1=200, param2= 20, minRadius=40, maxRadius=80)
    return circles[0]

    
    
        

if __name__ == "__main__":
    # Maak een nieuw venster
    cv2.namedWindow("Video Feed")
    
    # Stel de muiscallback functie in op get_position
    cv2.setMouseCallback("Video Feed", get_position)
    #######
    
    # setup webcam feed 
    cap = cv2.VideoCapture(0)  # Change this line to capture video from webcam
    
    kernel = np.ones((3,3), np.uint8)
    
    # # frame = cv2.imread('/Users/vojtadeconinck/Downloads/python-project/Labyrinth.jpeg')
    # ret, foto = cap.read()
    foto = cv2.imread("/Users/vojtadeconinck/Downloads/python-project/Schuinefoto.jpg")

    # # Converteer de afbeelding naar grijstinten
    # gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)

    # # Binariseer de afbeelding (maak de paden wit en de muren zwart)
    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # # Maak een masker van de muren
    # mask = binary == 0

    # # Maak een kopie van de originele afbeelding
    # img2 = foto.copy()

    # # Maak de muren rood in de kopie
    # img2[mask] = [0, 0, 255]

    # foto = img2

    frame = cv2.GaussianBlur(foto, (5,5), 0)
  

    while test_hue is None:
        
        # _, frame = cap.read()
        cirkel_coord = balldetection(frame)
        # cv2.circle(frame, (int(cirkel_coord[0]), int(cirkel_coord[1])), int(r), (0,0,0), 3)
        
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
    
    # Assuming red_mask is your image
    coords = cv2.findNonZero(red_mask)
    x, y, w, h = cv2.boundingRect(coords)
    print('coords',x,y,h,w)
    # Crop the image to the found coordinates

    crop = red_mask[y:y+h, x:x+w]

    # crop = cv2.resize(crop, None, fx = 0.5, fy = 0.5)
    
    # Skeletonize the image
    padcrop = np.logical_not(crop)
    padskelet = skeletonize(padcrop)
    padskelet_int = (padskelet.astype(np.uint8))*255
    
    padskelet_final = cv2.dilate(padskelet_int, kernel, iterations=1)
    
    # cv2.imshow("Video Feed",     padskelet)
    #Roep functie op waar we begin en einde van de maze bepalen
    cv2.setMouseCallback("Video Feed", pos_start)
    
    # Bepaal de grootte van de tekst
    (text_width, text_height) = cv2.getTextSize("KLIK OP START", cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0]
    
    # Bepaal de positie van de tekst
    text_x = int(len(crop[0])*0.4)
    text_y = len(crop)//2

    color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    # Teken een zwarte rechthoek achter de tekst
    cv2.rectangle(color_frame, (text_x - 5, text_y + 5), (text_x + text_width + 5, text_y - text_height - 5), (0, 0, 255), -1)
    # Teken de tekst over de rechthoek
    cv2.putText(color_frame, "KLIK OP STARTPUNT", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    
    randpunten =[[0,0],[0,w],[h,0],[h,w]]

    print(randpunten)
    # randpunten =[(x,y),(x+w,y)]
    for i in range(len(randpunten)):
            cv2.circle(color_frame, (randpunten[i][1], randpunten[i][0]), 15, (255,0,0), 3) 
    print('shape',crop.shape)
    
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bit_crop = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
    skelet_crop = skeletonize(bit_crop)
    randen_groen = find_closest_skeleton_point_with_kdtree(randpunten, crop)
    randen_groen = [list(i) for i in randen_groen]
    

    for i in range(len(randen_groen)):
        cv2.circle(color_frame, (randen_groen[i][1], randen_groen[i][0]), 20, (0,255,0), 3) 
    randpunten = np.array(randpunten, dtype=np.float32)
    randen_groen = np.array(randen_groen, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(randen_groen, randpunten)
    result = cv2.warpPerspective(color_frame, matrix, (len(color_frame[0]), len(color_frame)))
    

    cv2.imshow("Video Feed", result)
    cv2.waitKey(100000)

    
    
    while start is None:
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", color_frame)
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(start)
    (text_width, text_height) = cv2.getTextSize("KLIK OP EINDPUNT", cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0]
    cv2.rectangle(color_frame, (text_x - 5, text_y + 5), (text_x + text_width + 5, text_y - text_height - 5), (0, 0, 255), -1)
    cv2.putText(color_frame, "KLIK OP EINDPUNT", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)

    cv2.setMouseCallback("Video Feed", pos_end)

    while end is None:
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", color_frame)
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(end)
    
    start = find_closest_skeleton_point_with_kdtree([start], padskelet_final)[0] #start projecteren op padskelet
    end = find_closest_skeleton_point_with_kdtree([end], padskelet_final)[0] #end projecteren op padskelet
    
    
    start_time = time.time()
    distances = bf.breadth_first(padskelet_final, start, end)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    
    path = bf.print_shortest_path(distances, start, end)
    checkpoints = path.copy()


    color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)


    for i in range(len(path) - 1):
        x , y = path[i]
        point1 = (int(y), int(x))
        x , y = path[i + 1]
        point2 = (int(y), int(x))
        cv2.line(color_frame, point1, point2, (0, 0, 255), 2)
        cv2.circle(color_frame, point1, 4, (0,255,0), 1)

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
    cv2.waitKey(100)
    
    Lijst_cirkels = []
    
    start_time2 = time.time()
    time_overload = 0
    while checkpoints:
        _, frame = cap.read()
        cirkel_coord = balldetection(frame)
        Lijst_cirkels.append(cirkel_coord)
        cv2.circle(frame, (int(cirkel_coord[0]), int(cirkel_coord[1])), int(r), (0,0,0), 3)
        
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", frame)
        #to quit
        if abs(cirkel_coord[0] - checkpoints[0][0] < 10 and cirkel_coord[1]- checkpoints[0][1] < 10):
            checkpoints.pop(0)
            
        if time.time() - start_time2 > time_overload:
            pid.PIDcontroller(Lijst_cirkels[-1], checkpoints)
            time_overload += 3000
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    #Voor centreren: centerlines --> Lloris zegt: vindt een vorm en pakt dan het midden van de vorm git config --global --edit

