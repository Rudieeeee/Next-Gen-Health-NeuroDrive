/**************************************************************
* Sender.ino
* Bluetooth central device (EMG).
* Read data from analog port
* Send a byte to the peripheral device (Host computer) every second
**************************************************************/
#include <ArduinoBLE.h>

// ADC settings
// Set resolution to maximum of arduino Giga. Default is 10bit
int resolutionBits = 16; 
int analogPin = A0;
int updateTime = 500; // Sending time interval



void setup() {
  // RGB led
  pinMode(86, OUTPUT);
  pinMode(87, OUTPUT);
  pinMode(88, OUTPUT);

  // Serial communication
  Serial.begin(115200);
  delay(1000);
  Serial.println("Sender.ino");

  analogReadResolution(resolutionBits); 
  
  // Bluetooth initialization
  if (!BLE.begin()) {
    Serial.println("Starting Bluetooth Low Energy module failed!");
    while (1);
  }
  Serial.println("******************************************************");
  Serial.println("Bluetooth ready");
  Serial.print("Arduiono BT Address : ");
  Serial.println(BLE.address());
}
  
// Control Red led
void RedLed(bool state){
  if (state) {
    (GPIOI->ODR) = 0x0000;
  }
  else {
    (GPIOI->ODR) = 0x1000;
  }
}

// Control Green led
void GreenLed(bool state){
  if (state) {
    (GPIOJ->ODR) = 0x0000;
  }
  else {
    (GPIOJ->ODR) = 0x2000;
  }
}

// Control Blue led
void BlueLed(bool state){
  if (state) {
    (GPIOE->ODR) = 0x0000;
  }
  else {
    (GPIOE->ODR) = 0x0008;
  }
}

int readADC() {
  double average;
  int sensorValue = analogRead(analogPin); // read input on A0 analog pin
  int count = 1;
  // Average. Say the samplerate is 100 kHz, and update 0.5 sec
  // then we could in theory sample 50k times.
  int N = 100;
  while (count < N) {
    sensorValue = sensorValue + analogRead(A0);
    count = count + 1;
  }
  // Simply trunc, ok if average is large
  average = (int)(sensorValue/N);
  return average;
}

void loop() {
  // Red means no connection:
  BlueLed(false);
  GreenLed(false);
  RedLed(true);
  
  // Searching for Arduino Computer
  Serial.println("Searching for Computer ...");
  BLE.scanForName("NextComputer");
  // BLue means Computer found
  RedLed(false);
  BlueLed(true);

  // Create Bluetooth device
  Serial.println("Create BLEDevice");
  BLEDevice Computer = BLE.available();
  while (not Computer) {
    Computer = BLE.available();
  }
  Serial.println("");

  if (Computer) {
    // Show properties
    Serial.print("Found ");
    Serial.print(Computer.address());
    Serial.print(" '");
    Serial.print(Computer.localName());
    Serial.print("' ");
    Serial.print(Computer.advertisedServiceUuid());
    Serial.println();

    // Try to connect 10 times, than give up
    int count = 0;  
    while (!Computer.connect() & (count < 10) ) {
      Serial.print("Connecting. Attempt: ");
      Serial.println(count);
      count = count + 1;
    }
    if (count == 10) {
      Serial.println("Failed");
      return;
    }
    // As long as it is connected, send data
    if (Computer.connected()) {
      // Green means bluetooth connection succesfully made. Led will blink
      BlueLed(false);
      GreenLed(true);
      // Get characteristics from Computer 
      Computer.discoverAttributes();
      // Create characteristic
      BLECharacteristic writeCharacteristic = Computer.characteristic("51FF12BB-3ED8-46E5-B4F9-D64E2FEC021B");
      // Using millis() to time, delay does not work well in communication
      // https://forum.arduino.cc/t/blecharacteristic-writevalue-not-working-for-peripheral/1052664
      unsigned long previousMillis = 0;  // last time LED was updated
      bool ledState = false; // to toggle led
      while (Computer.connected()) {
	unsigned long currentMillis = millis();
	if (currentMillis - previousMillis >= updateTime) {
	  previousMillis = currentMillis;
	  // Read ADC converter
	  int value = readADC();
	  // Convert int (16 bit) to two bytes, send them seperately
	  Serial.print("Sending data: ");
	  Serial.print(value);
	  Serial.print(" , ");
	  Serial.print(highByte(value));
	  Serial.print(" , ");
	  Serial.println(lowByte(value));
	  // First send a 0 to notify computer new integer is arriving
	  writeCharacteristic.writeValue((byte)0x00);
	  // Than send two bytes
	  writeCharacteristic.writeValue(highByte(value));
	  writeCharacteristic.writeValue(lowByte(value));
	  GreenLed(ledState);
	  ledState = !ledState;
	}
      }
    }
    Computer.disconnect();
  }
  
  Serial.println("Computer not available");
  // Wait for starting next discovery 
  delay(100);
}

