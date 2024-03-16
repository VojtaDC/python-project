
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

ser = serial.Serial('/dev/tty.usbmodem1401', 9600, timeout=1)


def PIDcontroller(coordinaten, checkpoints):
    
    command_a_plus = f'A{150}\n'.encode()
    command_a_neutraal = f'A{118}\n'.encode()
    command_a_min = f'A{84}\n'.encode()
    command_b_plus = f'B{170}\n'.encode()
    command_b_neutraal = f'B{140}\n'.encode()
    command_b_min = f'B{110}\n'.encode()
    
    error_x = checkpoints[0][1] - coordinaten[0]
    print('error_x = ', error_x)

    error_y = checkpoints[0][0] - coordinaten[1]
    print('error_y = ', error_y)
    
    print('PID')
    if error_x > 20:
        # ser.write(command_b_plus)
        ser.write(('B-' + '\n').encode())
        print(command_b_plus)
    if error_x < -20:
        # ser.write(command_b_min)
        ser.write(('B+' + '\n').encode())
        print(command_b_min)

    if error_y > 20:
        # ser.write(command_a_plus)
        ser.write(('A+' + '\n').encode())
        print(command_a_plus)
    if error_y < -20:
        # ser.write(command_a_min)

        ser.write(('A-' + '\n').encode())
        print(command_a_min)

    return checkpoints
#118 en 140 

#MAX 150 172
#MIN 84 110