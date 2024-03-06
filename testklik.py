import cv2
import numpy as np

# Functie die wordt aangeroepen bij muisklik
def get_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Als er op de linkermuisknop wordt geklikt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converteer het frame naar HSV
        color = hsv[y, x]  # Haal de kleur op van de pixel waarop is geklikt
        return color
        print(f'Je hebt geklikt op positie: ({x}, {y}), kleur in HSV: {color}')  # Print de positie en kleur

# Maak een nieuw venster
cv2.namedWindow('Video Feed')

# Stel de muiscallback functie in op get_position
cv2.setMouseCallback('Video Feed', get_position)

# Open de videofeed
cap = cv2.VideoCapture(0)  # 0 is de standaard camera

while True:
    # Lees een frame uit de videofeed
    ret, frame = cap.read()

    # Als het frame succesvol is gelezen, toon het dan
    if ret:
        cv2.imshow('Video Feed', frame)

    # Stop de lus als 'q' wordt ingedrukt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sluit de videofeed en vernietig alle vensters
cap.release()
cv2.destroyAllWindows()