# Code to read and display ADC A0 over a bluetooth connection

- Sender.ino : Code for Arduino Giga that reads ADC and sends data in
  bytes to computer or second Arduino Giga
- Receiver.ino : Code for Arduino Giga that reads data from the
  Arduino Giga running Sender.ino. Use for example the built-in Serial Plotter of
  the Arduino IDE to display the data
- DataServer.py : Daemon running on computer that waits for bluetooth
  connection and saves received data to EMGData.csv
- BTGraph.py : Python code that continuously reads last 20 datapoints from
  EMGData.csv and displays a graph

