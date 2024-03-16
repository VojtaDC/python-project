import serial
import time

ser = serial.Serial('/dev/tty.usbmodem11301', 9600)  # Vervang 'COM3' door de juiste poort

def send_command(command):
    ser.write((command + '\n').encode())
    time.sleep(0.1)  # Een korte pauze om de Arduino de tijd te geven te reageren

try:
    while True:
        command = input("Voer A+, A-, B+ of B- in: ")
        if command in ["A+", "A-", "B+", "B-"]:
            send_command(command)
        else:
            print("Ongeldig commando.")
except KeyboardInterrupt:
    ser.close()  # Sluit de seriële poort wanneer het script wordt gestopt
#118 en 140 

#MAX 150 172
#MIN 84 110