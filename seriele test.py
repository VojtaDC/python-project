import serial
import time

# Verander dit naar de juiste COM-poort
ser = serial.Serial('/dev/tty.usbserial-AQ0197Y7', 9600, timeout=1) 

def send_servo_positions(servo1_pos, servo2_pos):
    # Bouw de commando string op
    command = f'A{servo1_pos}\n'.encode()
    # Verstuur de commando string via UART
    ser.write(command)
    # Print het verstuurd commando voor bevestigingd
    print(f"Verstuurd: {command.decode().strip()}")

try:
    # Voorbeeld gebruik van de functie
    
    servo1_pos = 200  # Positie voor de eerste servomotor
    servo2_pos = 200  # Positie voor de tweede servomotor
    send_servo_positions(servo1_pos, servo2_pos)

    # Zorg ervoor dat je enige vertraging geeft om te zorgen dat het bericht is ontvangen en verwerkt
    time.sleep(2)

except Exception as e:
    print(f"Er is een fout opgetreden: {e}")

finally:
    ser.close()  # Vergeet niet om de seriële poort te sluiten!
