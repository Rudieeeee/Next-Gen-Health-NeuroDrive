/**************************************************************
* Sender.ino
* Bluetooth central device (EMG).
* Send a byte to the peripheral device (Wheelchair) every second
**************************************************************/
#include <ArduinoBLE.h>

const char* WheelChairAddress = "a8:61:0a:1d:67:1b"; // Bluetooth Receiver (Wheelchair)
int blinkTime = 1000; // Sending time interval

void setup() {
  // RGB led
  pinMode(86, OUTPUT);
  pinMode(87, OUTPUT);
  pinMode(88, OUTPUT);

  // Serial communication
  Serial.begin(115200);
  delay(1000);
  Serial.println("Sender.ino");

  // Bluetooth initialization
  if (!BLE.begin()) {
    Serial.println("Starting Bluetooth Low Energy module failed!");
    while (1);
  }
  Serial.println("Bluetooth ready");
  Serial.println("******************************************************");
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

void loop() {
  // Red means no connection:
  BlueLed(false);
  GreenLed(false);
  RedLed(true);
  
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
  RedLed(false);
  BlueLed(true);

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
    // As long as it is connected, send data
    if (Wheelchair.connected()) {
      // Green means bluetooth connection succesfully made. Led will blink
      BlueLed(false);
      GreenLed(true);
      // Get characteristics from Wheelchair 
      Wheelchair.discoverAttributes();
      // Create characteristic
      BLECharacteristic writeCharacteristic = Wheelchair.characteristic("0001");

      // Using millis() to time, delay does not work well in communication
      // https://forum.arduino.cc/t/blecharacteristic-writevalue-not-working-for-peripheral/1052664
      unsigned long previousMillis = 0;  // last time LED was updated
      bool ledState = false; // to toggle led
      while (Wheelchair.connected()) {
	unsigned long currentMillis = millis();
	if (currentMillis - previousMillis >= blinkTime) {
	  previousMillis = currentMillis;
	  Serial.println("Sending data");
	  writeCharacteristic.writeValue((byte)0x01);
	  GreenLed(ledState);
	  ledState = !ledState;
	}
      }
    }
    Wheelchair.disconnect();
  }
  
  Serial.println("Wheelchair not available");
  // Wait for starting next discovery 
  delay(100);
}

