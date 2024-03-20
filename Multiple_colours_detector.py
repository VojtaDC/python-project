
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
from multiprocessing import Process, Queue

# import serieleTest as st

hue_threshold = None
start = None
end = None
crop = None
x = None
y = None
w = None
h = None
kernel = np.ones((3,3), np.uint8)


def get_position(event, x, y, flags, color): # Functie die wordt aangeroepen bij muisklik
    global hue_threshold
    if event == cv2.EVENT_LBUTTONDOWN:  # Als er op de linkermuisknop wordt geklikt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converteer het frame naar HSV
        color = hsv[y, x]  # Haal de kleur op van de pixel waarop is geklikt
        global hue_threshold
        hue_threshold = color[0]
        # print(color)

def pos_start(event, x, y, flags, color): #Geeft startcoordinaten bij muisklik
    global start
    if event == cv2.EVENT_LBUTTONDOWN:  # Als er op de linkermuisknop wordt geklikt
        global start
        start = (y, x)


def pos_end(event, x, y, flags, color): #Geeft eindcoordinaten bij muisklik
    global end
    if event == cv2.EVENT_LBUTTONDOWN:
        global end
        end = (y, x)

def color_rangefilter(hue_threshold, saturation_threshold, value_threshold): #Geeft kleurinteval terug op basis van HSV_thresholds
    # Pas de saturation en value bereiken aan om zwarte en zeer donkere kleuren uit te sluiten
    if hue_threshold > 165:
        lower_red = np.array([0, saturation_threshold, value_threshold]) 
        upper_red = np.array([hue_threshold-165, 255, 255])
        
        lower_red2 = np.array([hue_threshold-15, saturation_threshold, value_threshold]) 
        upper_red2 = np.array([180, 255, 255])
        
        return lower_red, upper_red, lower_red2, upper_red2
    
    # Range for upper range
    elif hue_threshold < 15:
        lower_red = np.array([0, saturation_threshold, value_threshold]) 
        upper_red = np.array([hue_threshold+15, 255, 255])
        
        lower_red2 = np.array([180-hue_threshold, saturation_threshold, value_threshold]) 
        upper_red2 = np.array([180, 255, 255])
        
        return lower_red, upper_red, lower_red2, upper_red2
    else:
        lower_red = np.array([hue_threshold-15, saturation_threshold, value_threshold]) 
        upper_red = np.array([hue_threshold+15, 255, 255])
        
        return lower_red, upper_red, None, None

    
def find_closest_skeleton_point_with_kdtree(path, padskelet): #Projectie van een lijst met punten op het dichtsbijzijnde padpunt
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

def Bballdetection(frame): #Geef lijst met alle cirkels op een frame
    coordinates = []
    newimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(newimg, cv2.HOUGH_GRADIENT, 0.1, 20000, param1=31, param2= 43, minRadius=8, maxRadius=25)
    if circles is not None:
        for x, y, r in circles[0]:
            # cv2.circle(frame, (int(x), int(y)), int(r), (0,0,0), 3)
            coordinates.append((x,y,r))
        # print('aaaaaaaaa')
        return coordinates
    
    else:
        print("none")
        return None
    
def red_mask(frame, color_range, erode1, dilate, erode2): 
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Converteert van BGR naar HSV
    
    red_mask = cv2.inRange(hsv_frame, color_range[0], color_range[1])
    if color_range[2] is not None: #Meerdere tinten van de geselecteerde kleur toevoegen
        mask2 = cv2.inRange(hsv_frame, color_range[2], color_range[3])
        red_mask += mask2
        
    
    
    red_mask = cv2.erode(red_mask, kernel, iterations= erode1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations= dilate)
    red_mask = cv2.erode(red_mask, kernel, iterations= erode2)
    return red_mask
def to_bitmap(frame):
    bitmap = np.where(frame == 255, 1, 0) #Alle waarden die 255 zijn worden vervangen door 1, rest blijft onveranderd
    return bitmap

def to_inverse_bitmap(frame):
    inverse_bitmap = np.where(frame == 255, 0, 1) #Alle waarden die 255 zijn worden vervangen door 0, andere pixels worden 1
    return inverse_bitmap
import numpy as np
import cv2
from skimage.morphology import skeletonize

def skeletonize_frame(frame):
    # Skeletonize the frame
    skeleton = skeletonize(frame == 1)

    # Convert the frame to uint8 and multiply by 255
    frame = (skeleton.astype(np.uint8) * 255)

    # Define the structuring element for the dilate operation
    kernel = np.ones((3,3),np.uint8)

    # Perform the dilate operation
    frame = cv2.dilate(frame, kernel, iterations = 1)

    return frame
def crop(red_mask): 
    global x
    global y
    global w
    global h
    
    coords = cv2.findNonZero(red_mask)
    x, y, w, h = cv2.boundingRect(coords)
    crop = red_mask[y:y+h, x:x+w]
    
    # crop = cv2.resize(crop, None, fx = 0.5, fy = 0.5)
    return crop

def red_converter(foto):
    # # Converteer de afbeelding naar grijstinten
    gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)

    # # Binariseer de afbeelding (maak de paden wit en de muren zwart)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # # Maak een masker van de muren
    mask = binary == 0

    # # Maak een kopie van de originele afbeelding
    red_foto = foto.copy()

    # # Maak de muren rood in de kopie
    red_foto[mask] = [0, 0, 255]
    
    return red_foto
    
def circle_plotter(frame, punten, dikte, straal, BGR):
    for punt in punten:
        cv2.circle(frame, punt, radius= straal, color= BGR, thickness= dikte)
    return frame
def bereken_en_update_pad(start, end, padskelet, result_queue):
    projectie_start = find_closest_skeleton_point_with_kdtree([start], padskelet)[0]
    distances = bf.breadth_first(padskelet, projectie_start, end)
    path = bf.print_shortest_path(distances, projectie_start, end)
    checkpoints = [tuple(int(x_t) for x_t in tup) for tup in path]
    result_queue.put(checkpoints)  # Zet het resultaat in de queue
    

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


if __name__ == "__main__":
    result_queue = Queue()
    process = None
    
    # Maak een nieuw venster
    cv2.namedWindow("Video Feed")
    # Stel de muiscallback functie in op get_position
    cv2.setMouseCallback("Video Feed", get_position)
    cap = cv2.VideoCapture(0)  #Webcam feed
    
    #_, foto = cap.read() #_ is boolean die aangeeft of frame succesvol gelezen is
    
    # # frame = cv2.imread('/Users/vojtadeconinck/Downloads/python-project/Labyrinth.jpeg')

    # frame = cv2.GaussianBlur(foto, (5,5), 0)
  

    while hue_threshold is None:
        
        _, frame = cap.read()
        # cirkel_coord = balldetection(frame)
        # cv2.circle(frame, (int(cirkel_coord[0]), int(cirkel_coord[1])), int(r), (0,0,0), 3)
        
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", frame)
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('ja')
    
    kleur_intervallen = color_rangefilter(hue_threshold, 70, 70)
    print("kleur intervallen =", kleur_intervallen)
    
    red_mask_startbeeld = red_mask(frame, kleur_intervallen, erode1 = 1, dilate=6, erode2=2)
    
    crop = crop(red_mask_startbeeld) #cropt met globale x,y,h,w
    
    inverse_bitcrop = to_inverse_bitmap(crop) #255 wordt 0 en 0 wordt 1
    
    padskelet = skeletonize_frame(inverse_bitcrop) #Maak pad dunner --> skelet
    randhoogte = int(len(padskelet)/25)
    padskelet[:randhoogte,:] = 0
    padskelet[-randhoogte:,:] = 0
    randbreedte = int(len(padskelet[0])/25)
    padskelet[:, :randbreedte] = 0
    padskelet[:,-randbreedte:] = 0
    
    #Roep functie op waar we begin en einde van de maze bepalen
    cv2.setMouseCallback("Video Feed", pos_start)
    
    # Bepaal de grootte van de tekst
    (text_width, text_height) = cv2.getTextSize("KLIK OP STARTPUNT", cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0]
    
    # Bepaal de positie van de tekst
    text_x = int(len(crop[0])*0.3)
    text_y = len(crop)//2

    color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    # Teken een zwarte rechthoek achter de tekst
    cv2.rectangle(color_frame, (text_x - 5, text_y + 5), (text_x + text_width, text_y - text_height - 5), (0, 0, 255), -1)
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
   
    while start is None:
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", color_frame)
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("start= ", start)
    (text_width, text_height) = cv2.getTextSize(" KLIK OP EINDPUNT", cv2.FONT_HERSHEY_SIMPLEX, 1, 5)[0]
    cv2.rectangle(color_frame, (text_x - 5, text_y + 5), (text_x + text_width + 15, text_y - text_height - 5), (0, 0, 255), -1)
    cv2.putText(color_frame, " KLIK OP EINDPUNT", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)

    cv2.setMouseCallback("Video Feed", pos_end)

    while end is None:
        #cv2.imshow("Frame", frame)
        cv2.imshow("Video Feed", color_frame)
        #to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        
    print("end =", end)
    
    start = find_closest_skeleton_point_with_kdtree([start], padskelet)[0] #start projecteren op padskelet
    end = find_closest_skeleton_point_with_kdtree([end], padskelet)[0] #end projecteren op padskelet
    
    
    start_time = time.time()
    distances = bf.breadth_first(padskelet, start, end)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("voor shortest path")
    path = bf.print_shortest_path(distances, start, end)
    print("na shortest path")

    checkpoints_float = path.copy()
    checkpoints = [tuple(int(x) for x in tup) for tup in checkpoints_float]
    print('checkpoints= ', checkpoints)
    
   
    # print('checkpoints = ', checkpoints[0:5])

    # color_frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    # for i in range(len(path) - 1):
    #     x_p , y_p = path[i]
    #     point1 = (int(y_p), int(x_p))
    #     x_p , y_p = path[i + 1]
    #     point2 = (int(y_p), int(x_p))
    #     cv2.line(color_frame, point1, point2, (0, 0, 255), 2)
    #     cv2.circle(color_frame, point1, 4, (0,255,0), 1)

    print('nu wachten we')

    Lijst_cirkels = []
    
    start_time2 = time.time()
    time_overload = 3.0
    # while True:
    #     cv2.imshow("Video Feed", color_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    vorig_checkpoint = checkpoints[0]
    index = 0
    while checkpoints:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        crop = frame[y-10:y+h+10, x-10:x+w+10]
        # Stel dat checkpoints een lijst is van tuples, waarbij elke tuple de (x, y) coördinaten van een checkpoint bevat

        circle_plotter(crop, checkpoints,-1,5,(0, 255, 0))  # Teken een cirkel op elk checkpoint


        # Teken een lijn tussen elk paar opeenvolgende checkpoints
        for i in range(len(checkpoints) - 1):
            cv2.line(crop, checkpoints[i], checkpoints[i+1], color=(0, 255, 0), thickness=2)
        #Omzetten frame naar crop:
        
        red_mask_crop = red_mask(crop, kleur_intervallen, 1, 2, 0)

        

        randpunten_live =[[0,0],[0,w],[h,0],[h,w]]

        randen_groen = find_closest_skeleton_point_with_kdtree(randpunten_live, red_mask_crop) #ipv crop kan cropskelet_final ook gebruikt worden, maar dan geeft hij het getransformeerd beeld te ingezoomd weer, je zou dus dimensies daarvan nog moeten aanpassen.
        randen_groen = [list(i) for i in randen_groen]
        print(randpunten_live, randen_groen)

        #Cirkels tekenen:
        # for i in range(len(randpunten_live)):
        #     cv2.circle(crop, (randpunten_live[i][1], randpunten_live[i][0]), 15, (255,0,0), 3)
        # for i in range(len(randen_groen)):
        #     cv2.circle(crop, (randen_groen[i][1], randen_groen[i][0]), 5, (0,255,255), 5)

        randpunten_live = np.array(randpunten_live, dtype=np.float32)
        randen_groen = np.array(randen_groen, dtype=np.float32)


        Omgekeerde_randen_groen = np.array([[x_o,y_o] for y_o ,x_o in randen_groen])

        matrix = cv2.getPerspectiveTransform(Omgekeerde_randen_groen, Omgekeerde_randpunten)
        result_live = cv2.warpPerspective(crop, matrix, (len(crop[0]), len(crop)))

        # cv2.imshow("Video Feed", crop)
        Cirkels_coordinaat = Bballdetection(result_live)
        
        if Cirkels_coordinaat is not None:
            Cirkels_coordinaat[0] = tuple(int(x_u) for x_u in Cirkels_coordinaat[0])
            Lijst_cirkels.append(Cirkels_coordinaat[0])
            # print('CCCCC=',Cirkels_coordinaat[0], checkpoints[0] )
        
        
        # Lijst_cirkels.append(cirkel_coord)
        # cv2.circle(frame, (int(Lijst_cirkels[0][0]), int(Lijst_cirkels[0][1])), 10, (0,255,0), 2)
        if Lijst_cirkels:
            print(Lijst_cirkels[-1])
            straal = Lijst_cirkels[-1][2]
            
            cv2.circle(result_live, (Lijst_cirkels[-1][0], Lijst_cirkels[-1][1]), Lijst_cirkels[-1][2], (255,0,0), 3)
            print(abs(Lijst_cirkels[-1][0] - checkpoints[0][1]), abs(Lijst_cirkels[-1][1]- checkpoints[0][0]))
            print('lijst cirkels = ',Lijst_cirkels[-1], 'lijst checkpoints= ', checkpoints[0])
            
            if abs(Lijst_cirkels[-1][0] - checkpoints[0][0]) < 2*straal and abs(Lijst_cirkels[-1][1]- checkpoints[0][1]) < 2*straal:
                checkpoints.pop(0)
                print('checkpoint gepopt')
            if int(time.time()*10)%10 == 0:
                pid.PIDcontroller(Lijst_cirkels[-1][:2], checkpoints, len(result_live))
                print("Opgeroepen PID")
                time_overload += 3.0
                
            if int(time.time()) % 1 == 0 and (process is None or not process.is_alive()): #Elke 5 seconden checken 
                print("chicken")
                if process is not None:
                    # Haal resultaten op als het proces klaar is
                    if not result_queue.empty():
                        checkpoints2 = result_queue.get()
                     # Dit blokkeert totdat er een resultaat is
                    print("Kortste pad bijgewerkt")
                if checkpoints[0] == vorig_checkpoint:
                    # Start een nieuw proces
                    x_l, y_l, r_l = Lijst_cirkels[-1]
                    start = (y_l, x_l)
                    process = Process(target=bereken_en_update_pad, args=(start, end, padskelet, result_queue))
                    process.start()
                else:
                    vorig_checkpoint = checkpoints[0]
                index +=1
                if (index%15) == 0:
                    checkpoints = checkpoints2
                    print("Kortste pad initialisatie")
                    

                
        cv2.imshow("Video Feed", result_live)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print("hier")
        #onderstaande moet blijven:

        #to quit
        
        

        

    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    
    #Voor centreren: centerlines --> Lloris zegt: vindt een vorm en pakt dan het midden van de vorm git config --global --edit

