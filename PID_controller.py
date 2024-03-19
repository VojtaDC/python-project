import serial
import time

ser = serial.Serial('/dev/tty.usbmodem11101', 9600, timeout=1)

def calculate_servo_position(error, min_pos, max_pos, min_error, max_error):
    if abs(error) < min_error:
        return (min_pos + max_pos) // 2  # Neutraal als de fout klein is
    else:
        # Schaal de fout binnen het servo bereik
        scale = (max_pos - min_pos) / (max_error - min_error)
        position = (min_pos + max_pos) // 2 + scale * min(max(error, -max_error), max_error)
        return int(constrain(position, min_pos, max_pos))

def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def PIDcontroller(coordinaten, checkpoints, size):
    min_error = size / 25
    max_error = size /2
    
    error_x = coordinaten[0] - checkpoints[0][0]
    print('error_x = ', error_x)

    error_y = checkpoints[0][1] - coordinaten[1]
    print('error_y = ', error_y)
    
    # Bereken servo posities
    servo_a_pos = calculate_servo_position(error_y, 90, 150, min_error, max_error)
    servo_b_pos = calculate_servo_position(error_x, 112, 172, min_error, max_error)
    
    # Stuur commando's naar de servo's
    command_a = f'A{str(servo_a_pos).zfill(3)}\n'.encode()
    command_b = f'B{str(servo_b_pos).zfill(3)}\n'.encode()

    ser.write(command_a)
    ser.write(command_b)
    print(f'Servo A naar {servo_a_pos}, Servo B naar {servo_b_pos}')

    return checkpoints
