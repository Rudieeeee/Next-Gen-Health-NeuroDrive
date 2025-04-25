# Arduino code for simple wheelchair control using bluetooth

Both single button and bluetooth digital control of R-NET operated
electric wheelchair through an IOM module.

- Wheelchair/Wheelchair.ino : Peripherial / server. Sends IOM command on button
  press or receiving byte from EMG.ino or EEG.ino
- EMG/EMG.ino : Central / client, reads analog in and sends byte to
  Wheelchair.ino when the algorithm decides the EMG is signal is
  sufficiently high
- EEG/EEG.ino : Central/ client. Is connected to notebook and reads over
  serial port if a signal needs to be sent to Wheelchair.ino
- EEGPython : Scripts for reading Unicorn EEG helmet and serial
  communication with Arduino running EEG.ino
  
  
