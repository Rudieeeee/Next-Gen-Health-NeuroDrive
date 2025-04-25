#####################################################################
# EEGExample.py
#
# Version 1.0. Leon Abelmann, Jan 2025
#
# Send command to Arduino running EEG.ino
#
# Requires pyserial
#####################################################################

import serial
import time

# Use this to determine the port on which the Arduino is on
# from serial.tools import list_ports
# port = list(list_ports.comports())
# for p in port:
#     print(p.device)

arduino = serial.Serial(port='COM7' ,#'/dev/cu.usbmodem2101',
                        baudrate=115200, timeout=.1)
# https://stackoverflow.com/questions/61166544/readline-in-pyserial-sometimes-captures-incomplete-values-being-streamed-from

def write_read(x):
    arduino.write(bytes(x,   'utf-8'))
    time.sleep(0.05)    
    val = arduino.readline()    # read complete line from serial output
    while not '\\n'in str(val): # check if full data is received. 
        # This loop is entered only if serial read value doesn't contain \n
        # which indicates end of a sentence. 
        # str(val) - val is byte where string operation to check `\\n` 
        # can't be performed
        time.sleep(.001)                # delay of 1ms 
        temp = arduino.readline()       # check for serial output.
        if not not temp.decode():       # if temp is not empty.
            val = (val.decode()+temp.decode()).encode()
            # requrired to decode, sum, then encode because
            # long values might require multiple passes
    val = val.decode() # decoding from bytes
    val = val.strip()  # stripping leading and trailing spaces.
    return val

print("EEGExample. Type any number you like, but 42 does the trick");

while True:
    num = input("Enter a number: ")
    value   = write_read(num)
    print(value)
