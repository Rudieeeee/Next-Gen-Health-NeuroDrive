/**************************************************************
 * EMG.ino
 *
 * Version 1.2 Leon Abelmann, Jan 2025. Implemented GigaLED libarary
 * 
 * Version 1.1 Separated out serveWheelchair to accomodate for EEG
 *
 * Bluetooth central device (Client, in this case EMG).  Detects EMG
 * signal and sends a byte to the peripheral device (server, in this
 * case the wheelchair)
 **************************************************************/
#include "EMG.h"

// Settings
const char* WheelChairAddress = "a8:61:0a:38:64:62"; // Bluetooth Peripheral (Wheelchair)
int         delayTime = 250; // Time before new EMG is detected
double      emgThreshold = 0.3; // Volt, Manually chosen emg threshold

// Global variables
int         emgBufferSize = 10;
double      emgBuffer[10]; // Buffer storing 6 analog read values to add some stability to the led 

// RGB Led
GigaLED LED;

void setup() {

  // Serial communication
  Serial.begin(115200);
  delay(1000);
  Serial.println("EMG.ino");

  // Bluetooth initialization
  if (!BLE.begin()) {
    Serial.println("Starting Bluetooth Low Energy module failed!");
    while (1);
  }
  Serial.println("Bluetooth ready");
}

/**************************************************************************
* EMGDetection
* Check for EMG signal
**************************************************************************/
bool EMGDetect() {
  // EMG Measure code
  //Serial.print("Start EMG detection \n"); 

  bool emgDetected = false;
  int resolutionBits = 16; 
  analogReadResolution(resolutionBits); // Set resolution to maximum of arduino Giga. Default is 10bit
  int sensorValue = analogRead(A0); // read input on A0 analog pin

  double arduinoVoltage = sensorValue * 3.3 / (pow(2, resolutionBits)-1); // plot and check if sensorValue max is 2^16 or 2^15 (possible due to sign)
  for (int i = emgBufferSize-1; i > 0 ; i--) { emgBuffer[i]=emgBuffer[i-1]; }
  emgBuffer[0]=arduinoVoltage;

  double totalEmgBuf = 0;
  for (int i = 0; i < emgBufferSize; i++) { totalEmgBuf = totalEmgBuf + emgBuffer[i]; }
  double emgAvg = totalEmgBuf/emgBufferSize;

  if (emgAvg > emgThreshold) {
    Serial.print("EMG over threshold \n"); 
    emgDetected = true;
  }
  else {
    emgDetected = false;
  }

  Serial.print(arduinoVoltage);
  Serial.print(" ");
  Serial.print(emgAvg);
  Serial.print(" ");
  Serial.println(0);
  
  return emgDetected;
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
  int EMGcount = 0;
  while (Wheelchair.connected()) {
    unsigned long curTime = millis();
    if ((curTime - prevTime >= delayTime)) {
      prevTime = curTime; // Reset time counter
      // Turn off blue led, to indicate we are ready for next EMG
      LED.Green();
      // Check for EMG
      if (EMGDetect() & (EMGcount < 1) ) {
	LED.Blue();
	EMGcount = 10;
	// Notify Wheelchair and turn on blue led
	writeCharacteristic.writeValue((byte)0x01);
      }
      else {
	EMGcount = max(0,EMGcount - 1) ;
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
