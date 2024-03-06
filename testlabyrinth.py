import cv2
import numpy as np

# Lees de afbeelding
img = cv2.imread('/Users/vojtadeconinck/Downloads/Labyrinth.jpg')

# Converteer de afbeelding naar grijstinten
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binariseer de afbeelding om een labyrint te maken met witte achtergrond en rode muren
# Stel een drempelwaarde in. Alle pixels met een waarde hoger dan de drempelwaarde worden wit (255) en de rest wordt rood (0)
_, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Toon de afbeelding
cv2.imshow('Labyrinth', binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()