/**************************************************************
* EEG.ino
*
* Version 1.0 Bluetooth central device 
*
* Waits for command on serial port, and sends byte over Bluetooth to
* Arduino running Wheelchair.ino (Bluetooth peripheral).
*
* Intended usage:
*
* An EEG helmet is connected to a notebook that runs detection
* software. The notebook is connected to the serial port of the
* Arduino. If the software on the notebook decides that an EEG signal
* is detected, it sends a code to t the serial port. The Arduino then
* sends a byte to the Arduion connected to the serial device (the
* bluetooth peripheral device (server)
**************************************************************/
#include "EEG.h"

// Settings
const char* WheelChairAddress = "a8:61:0a:38:64:62"; // Bluetooth Peripheral (Wheelchair)
int delayTime = 250; // Time before new EMG is detected

GigaLED LED;

void setup() {

  // Serial communication
  Serial.begin(115200);
  delay(1000);
  Serial.println("EEG.ino");

  // Bluetooth initialization
  if (!BLE.begin()) {
    Serial.println("Starting Bluetooth Low Energy module failed!");
    while (1);
  }
  Serial.println("Bluetooth ready");
}



/**************************************************************************
* EEGDetect
* Check serial port if an EEG threshold has been crossed
**************************************************************************/
bool EEGDetect() {
  // EMG Measure code
  //Serial.print("Start EMG detection \n"); 

  bool eegDetected = false;

  // Check serial port
  if (Serial.available() > 0) {
    int x = Serial.readString().toInt();
    // Send x back to sender
    Serial.println(x);
    // When x=42, raise flag
    eegDetected = (x == 42);
  }  
  return eegDetected;
}

void serveWheelchair (BLEDevice Wheelchair) {
  // Green means bluetooth connection succesfully made. Led will blink
  LED.Green();

  // Get characteristics from Wheelchair 
  Wheelchair.discoverAttributes();
  // Create characteristic
  BLECharacteristic writeCharacteristic = Wheelchair.characteristic("0001");

  // While the connection is on, send byte if EMG is detected
  // To avoid a crash, try every delayTime
  unsigned long prevTime = 0;  // last update time
  while (Wheelchair.connected()) {
    unsigned long curTime = millis();
    if ((curTime - prevTime >= delayTime)) {
      prevTime = curTime; // Reset time counter
      // Turn off blue led, to indicate we are ready for next EMG
      LED.Green();
      // Check for EEG
      if (EEGDetect()) {
	// Turn on blue led
	LED.Blue();
	// Notify Wheelchair
	writeCharacteristic.writeValue((byte)0x01);
	prevTime = millis(); // Wait extra to show led
      }
    }
  }
}

void loop() {
  // Red means no connection:
  LED.Red();
  // Searching for Arduino Wheelchair
  Serial.println("Searching for Wheelchair ...");
  if (BLE.scanForAddress(WheelChairAddress)) {
    Serial.println("Wheelchair address found");
  }
  else {
    Serial.println("Wheelchair address not found, trying name");
    if (BLE.scanForName("WheelChair")) {
      Serial.println("Wheelchair name found");
    }
    else {
      Serial.println("Wheelchair not found. Try again ...");
    return;
    }
  }

  // BLue means Wheelchair found
  LED.Blue();
  // Create Bluetooth device
  Serial.println("Create BLEDevice");
  BLEDevice Wheelchair = BLE.available();
  while (not Wheelchair) {
    Wheelchair = BLE.available();
  }

  if (Wheelchair) {
    // Show properties
    Serial.print("Found ");
    Serial.print(Wheelchair.address());
    Serial.print(" '");
    Serial.print(Wheelchair.localName());
    Serial.print("' ");
    Serial.print(Wheelchair.advertisedServiceUuid());
    Serial.println();

    // Try to connect 10 times, than give up
    int count = 0;  
    while (!Wheelchair.connect() & (count < 10) ) {
      Serial.print("Connecting. Attempt: ");
      Serial.println(count);
      count = count + 1;
    }
    if (count == 10) {
      Serial.println("Failed");
      return;
    }
    
    // As long as it is connected, check for EMG signal and send byte
    if (Wheelchair.connected()) {
      serveWheelchair(Wheelchair);
    }
    Wheelchair.disconnect();
  } // if(Wheelchair)
  Serial.println("Wheelchair not available");
  // Wait before starting next discovery
  delay(100);
} // loop()
