
error_x = 0
error_y = 0
#prev_error_x = 0
#prev_error_y = 0
#integral_x = 0.0
#Ki_x = 0.5
#Ki_y = 0.5
#Kp_x = 4.0
#Kp_y = 4.0
#dutycycle_x = 0
#dutycycle_y = 0

import serial
import time

# ser = serial.Serial('/dev/tty.usbserial-AQ0197Y7', 9600, timeout=1)


def PIDcontroller(coordinaten, checkpoints):
    command_a_plus = f'A{255}\n'.encode()
    command_a_neutraal = f'A{128}\n'.encode()
    command_a_min = f'A{0}\n'.encode()
    command_b_plus = f'B{255}\n'.encode()
    command_b_neutraal = f'B{128}\n'.encode()
    command_b_min = f'B{0}\n'.encode()
    
    error_x = checkpoints[0][0] - coordinaten[0]

    error_y = checkpoints[0][1] - coordinaten[1]
    if error_x > 10:
        ser.write(command_a_plus)
    if error_x < -10:
        ser.write(command_a_min)
    
    if error_y > 10:
        ser.write(command_b_plus)
    if error_y < -10:
        ser.write(command_a_min)

    return checkpoints