#!/usr/bin/env python

# Unicorn2lsl streams data from a Unicorn Hybrid Black EEG system to LSL
#
# Copyright (C) 2022 Robert Oostenveld
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import serial
import struct
import string
import random
import numpy as np
from pylsl import StreamInfo, StreamOutlet

#device='/dev/tty.UN-20230623'
device = 'COM6'
blocksize=0.2
timeout=5
nchan=16
fsample=250

start_acq      = [0x61, 0x7C, 0x87]
stop_acq       = [0x63, 0x5C, 0xC5]
start_response = [0x00, 0x00, 0x00]
stop_response  = [0x00, 0x00, 0x00]
start_sequence = [0xC0, 0x00]
stop_sequence  = [0x0D, 0x0A]

try:
    s = serial.Serial(device, 115200, timeout=timeout)
    print("connected to serial port " + device)
except:
    raise RuntimeError("cannot connect to serial port " + device)

lsl_name    = 'Unicorn'
lsl_type    = 'EEG'
lsl_format  = 'float32'
lsl_id      = ''.join(random.choice(string.digits) for i in range(6))
                 
# create an outlet stream
info = StreamInfo(lsl_name, lsl_type, nchan, fsample, lsl_format, lsl_id)
outlet = StreamOutlet(info)

print('started LSL stream: name=%s, type=%s, id=%s' % (lsl_name, lsl_type, lsl_id))

# start the Unicorn data stream
s.write(start_acq)
    
response = s.read(3)
if response != b'\x00\x00\x00':
    raise RuntimeError("cannot start data stream")

print('started Unicorn')

try:
    while True:
        dat = np.zeros(nchan)
        
        # read one block of data from the serial port
        payload = s.read(45)
        #print(payload)
        # check the start and end bytes
        if payload[0:2] != b'\xC0\x00':
            raise RuntimeError("invalid packet")
        if payload[43:45] != b'\x0D\x0A':
            raise RuntimeError("invalid packet")
    
        battery = 100*float(payload[2] & 0x0F)/15
    
        eeg = np.zeros(8)
        for ch in range(0,8):
            # unpack as a big-endian 32 bit signed integer
            eegv = struct.unpack('>i', b'\x00' + payload[(3+ch*3):(6+ch*3)])[0]
            # apply two’s complement to the 32-bit signed integral value if the sign bit is set
            if (eegv & 0x00800000):
                eegv = eegv | 0xFF000000
            eeg[ch] = float(eegv) * 4500000. / 50331642.
    
        accel = np.zeros(3)
        # unpack as a little-endian 16 bit signed integer
        accel[0] = float(struct.unpack('<h', payload[27:29])[0]) / 4096.
        accel[1] = float(struct.unpack('<h', payload[29:31])[0]) / 4096.
        accel[2] = float(struct.unpack('<h', payload[31:33])[0]) / 4096.
    
        gyro = np.zeros(3)
        # unpack as a little-endian 16 bit signed integer
        gyro[0] = float(struct.unpack('<h', payload[27:29])[0]) / 32.8
        gyro[1] = float(struct.unpack('<h', payload[29:31])[0]) / 32.8
        gyro[2] = float(struct.unpack('<h', payload[31:33])[0]) / 32.8
    
        counter = struct.unpack('<L', payload[39:43])[0]
    
        # collect the data that will be sent to LSL
        dat[0:8]   = eeg
        dat[8:11]  = accel
        dat[11:14] = gyro
        dat[14]    = battery
        dat[15]    = counter
            
        # send the data to LSL
        outlet.push_sample(dat)

        if ((counter % fsample) == 0):
            print('received %d samples, battery %d %%' % (counter, battery))

except:
    print('closing')
    s.write(stop_acq)
    s.close()
    del outlet
    
