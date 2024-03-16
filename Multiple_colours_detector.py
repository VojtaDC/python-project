
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
import time
# import serieleTest as st

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

def color_ranges(test_hue):
    # Pas de saturation en value bereiken aan om zwarte en zeer donkere kleuren uit te sluiten
    saturation_threshold = 120
    value_threshold = 100

    if test_hue > 165:
        lower_red = np.array([0, saturation_threshold, value_threshold]) 
        upper_red = np.array([test_hue-165, 255, 255])
        
        lower_red2 = np.array([test_hue-15, saturation_threshold, value_threshold]) 
        upper_red2 = np.array([180, 255, 255])
        
        return lower_red, upper_red, lower_red2, upper_red2
    
    # Range for upper range
    elif test_hue < 15:
        lower_red = np.array([0, saturation_threshold, value_threshold]) 
        upper_red = np.array([test_hue+15, 255, 255])
        
        lower_red2 = np.array([180-test_hue, saturation_threshold, value_threshold]) 
        upper_red2 = np.array([180, 255, 255])
        
        return lower_red, upper_red, lower_red2, upper_red2
    else:
        lower_red = np.array([test_hue-15, saturation_threshold, value_threshold]) 
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

def Bballdetection(frame):
    coordinates = []
    newimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(newimg, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2= 40, minRadius=3, maxRadius=30)
    if circles is not None:
        for x, y, r in circles[0]:
            cv2.circle(frame, (int(x), int(y)), int(r), (0,0,0), 3)
            coordinates.append((x,y,r))
        # print('aaaaaaaaa')
        return coordinates
    
    else:
        print("none")
        return None
    
    
        

if __name__ == "__main__":
    # Maak een nieuw venster
    cv2.namedWindow("Video Feed")

    # Stel de muiscallback functie in op get_position
    cv2.setMouseCallback("Video Feed", get_position)
    #######
    
    # setup webcam feed 
    cap = cv2.VideoCapture(1)  # Change this line to capture video from webcam
    
    kernel = np.ones((3,3), np.uint8)
    
    # # frame = cv2.imread('/Users/vojtadeconinck/Downloads/python-project/Labyrinth.jpeg')
    ret, foto = cap.read()
    # foto = cv2.imread("/Users/vojtadeconinck/Downloads/python-project/Schuinefoto.jpg")

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

    # frame = cv2.GaussianBlur(foto, (5,5), 0)
  

    while test_hue is None:
        
        _, frame = cap.read()
        # cirkel_coord = balldetection(frame)
        # cv2.circle(frame, (int(cirkel_coord[0]), int(cirkel_coord[1])), int(r), (0,0,0), 3)
        
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", frame)
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('ja')
    x_ranges = color_ranges(test_hue)
    
    crop = None
    ## loop to continuously acquire frames from the webcam
    
    
       # _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    red_mask = cv2.inRange(hsv_frame, x_ranges[0], x_ranges[1])
    if x_ranges[2] is not None:
        mask2 = cv2.inRange(hsv_frame, x_ranges[2], x_ranges[3])
        red_mask += mask2
        
    red_mask = cv2.erode(red_mask, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=6)
    red_mask = cv2.erode(red_mask, kernel, iterations=2)
    
    # Assuming red_mask is your image
    coords = cv2.findNonZero(red_mask)
    x, y, w, h = cv2.boundingRect(coords)
    print('coordinatVOOR = ', x,y,w,h)
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
    (text_width, text_height) = cv2.getTextSize("KLIK OP STARTPUNT", cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0]
    
    # Bepaal de positie van de tekst
    text_x = int(len(crop[0])*0.4)
    text_y = len(crop)//2

    color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    # Teken een zwarte rechthoek achter de tekst
    cv2.rectangle(color_frame, (text_x - 5, text_y + 5), (text_x + text_width + 5, text_y - text_height - 5), (0, 0, 255), -1)
    # Teken de tekst over de rechthoek
    cv2.putText(color_frame, "KLIK OP STARTPUNT", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    
    randpunten =[[0,0],[0,w],[h,0],[h,w]]

    # bit_crop = np.logical_not(padcrop)
    
    # cropskelet = skeletonize(bit_crop)
    # cropskelet_int = (cropskelet.astype(np.uint8))*255
    # cropskelet_final = cv2.dilate(cropskelet_int, kernel, iterations=1)
    
    
    randen_groen = find_closest_skeleton_point_with_kdtree(randpunten, crop) #ipv crop kan cropskelet_final ook gebruikt worden, maar dan geeft hij het getransformeerd beeld te ingezoomd weer, je zou dus dimensies daarvan nog moeten aanpassen. 
    randen_groen = [list(i) for i in randen_groen]
    print(randpunten, randen_groen)
    
    #Cirkels tekenen:
    for i in range(len(randpunten)):
                cv2.circle(color_frame, (randpunten[i][1], randpunten[i][0]), 15, (255,0,0), 3) 
    for i in range(len(randen_groen)):
        cv2.circle(color_frame, (randen_groen[i][1], randen_groen[i][0]), 5, (0,255,255), 5) 
        
    randpunten = np.array(randpunten, dtype=np.float32)
    randen_groen = np.array(randen_groen, dtype=np.float32)

    Omgekeerde_randpunten = np.array([[x,y] for y ,x in randpunten])
    Omgekeerde_randen_groen = np.array([[x,y] for y ,x in randen_groen])
    
    start_time = time.time()

    matrix = cv2.getPerspectiveTransform(Omgekeerde_randen_groen, Omgekeerde_randpunten)
    result = cv2.warpPerspective(color_frame, matrix, (len(color_frame[0]), len(color_frame)))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for getPerspectiveTransform and warpPerspective: {elapsed_time} seconds")
   
    # cv2.imshow("Video Feed", result)
    # cv2.waitKey(100000)

    
    
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
    checkpoints_float = path.copy()
    checkpoints = [tuple(int(x) for x in tup) for tup in checkpoints_float]
    
    # print('checkpoints = ', checkpoints[0:5])

    color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)


    for i in range(len(path) - 1):
        x_p , y_p = path[i]
        point1 = (int(y_p), int(x_p))
        x_p , y_p = path[i + 1]
        point2 = (int(y_p), int(x_p))
        cv2.line(color_frame, point1, point2, (0, 0, 255), 2)
        cv2.circle(color_frame, point1, 4, (0,255,0), 1)


    
    print('nu wachten we')

    
    Lijst_cirkels = []
    
    start_time2 = time.time()
    time_overload = 3.0
    # while True:
    #     cv2.imshow("Video Feed", color_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    
    Lijst_cirkels = []
    while checkpoints:
        _, frame = cap.read()
        crop = frame[y-10:y+h+10, x-10:x+w+10]
        # Stel dat checkpoints een lijst is van tuples, waarbij elke tuple de (x, y) coördinaten van een checkpoint bevat


        # Teken een cirkel op elk checkpoint
        for checkpoint in checkpoints:
            cv2.circle(crop, checkpoint, radius=5, color=(0, 255, 0), thickness=-1)

        # Teken een lijn tussen elk paar opeenvolgende checkpoints
        for i in range(len(checkpoints) - 1):
            cv2.line(crop, checkpoints[i], checkpoints[i+1], color=(0, 255, 0), thickness=2)
        #Omzetten frame naar crop:
        hsv_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
        
        red_mask = cv2.inRange(hsv_frame, x_ranges[0], x_ranges[1])
        if x_ranges[2] is not None:
            mask2 = cv2.inRange(hsv_frame, x_ranges[2], x_ranges[3])
            red_mask += mask2
            
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        
        

        randpunten_live =[[0,0],[0,w],[h,0],[h,w]]

        randen_groen = find_closest_skeleton_point_with_kdtree(randpunten_live, red_mask) #ipv crop kan cropskelet_final ook gebruikt worden, maar dan geeft hij het getransformeerd beeld te ingezoomd weer, je zou dus dimensies daarvan nog moeten aanpassen.
        randen_groen = [list(i) for i in randen_groen]
        print(randpunten_live, randen_groen)

        #Cirkels tekenen:
        # for i in range(len(randpunten_live)):
        #     cv2.circle(crop, (randpunten_live[i][1], randpunten_live[i][0]), 15, (255,0,0), 3)
        # for i in range(len(randen_groen)):
        #     cv2.circle(crop, (randen_groen[i][1], randen_groen[i][0]), 5, (0,255,255), 5)

        randpunten_live = np.array(randpunten_live, dtype=np.float32)
        randen_groen = np.array(randen_groen, dtype=np.float32)


        Omgekeerde_randen_groen = np.array([[x,y] for y ,x in randen_groen])

        matrix = cv2.getPerspectiveTransform(Omgekeerde_randen_groen, Omgekeerde_randpunten)
        result_live = cv2.warpPerspective(crop, matrix, (len(crop[0]), len(crop)))

        # cv2.imshow("Video Feed", crop)
        Cirkels_coordinaat = Bballdetection(result_live)
        
        if Cirkels_coordinaat is not None:
            Cirkels_coordinaat[0] = tuple(int(x) for x in Cirkels_coordinaat[0])
            Lijst_cirkels.append(Cirkels_coordinaat[0])
            # print('CCCCC=',Cirkels_coordinaat[0], checkpoints[0] )
            
        # Lijst_cirkels.append(cirkel_coord)
        # cv2.circle(frame, (int(Lijst_cirkels[0][0]), int(Lijst_cirkels[0][1])), 10, (0,255,0), 2)
        if Lijst_cirkels is not None:
            print(Lijst_cirkels[-1])
            cv2.circle(result_live, (Lijst_cirkels[-1][0], Lijst_cirkels[-1][1]), Lijst_cirkels[-1][2], (255,0,0), 3)
        
        cv2.imshow("Video Feed", result_live)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print("hier")
        
        #onderstaande moet blijven:

        #to quit
        print(abs(Lijst_cirkels[-1][0] - checkpoints[0][1]), abs(Lijst_cirkels[-1][1]- checkpoints[0][0]))
        print('lijst cirkels = ',Lijst_cirkels[-1][0], 'lijst checkpoints= ', checkpoints[0][0])
        if abs(Lijst_cirkels[-1][0] - checkpoints[0][1]) < 10 and abs(Lijst_cirkels[-1][1]- checkpoints[0][0]) < 10:
            checkpoints.pop(0)
            print('checkpoint gepopt')

        if int(time.time()*10)%10 == 0:
            pid.PIDcontroller(Lijst_cirkels[-1][:2], checkpoints)
            print("Opgeroepen PID")
            time_overload += 3.0

        

    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    
    #Voor centreren: centerlines --> Lloris zegt: vindt een vorm en pakt dan het midden van de vorm git config --global --edit

