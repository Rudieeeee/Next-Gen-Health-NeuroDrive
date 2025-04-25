// -*- coding: iso-8859-1 -*-
#include <ArduinoBLE.h>

// LED control functions (using low-level register writes for Arduino Giga)
void RedLed(bool state) {
  if (state) {
    (GPIOI->ODR) = 0x0000;  // Turn Red LED ON
  } else {
    (GPIOI->ODR) = 0x1000;  // Turn Red LED OFF
  }
}

void GreenLed(bool state) {
  if (state) {
    (GPIOJ->ODR) = 0x0000;  // Turn Green LED ON
  } else {
    (GPIOJ->ODR) = 0x2000;  // Turn Green LED OFF
  }
}

void BlueLed(bool state) {
  if (state) {
    (GPIOE->ODR) = 0x0000;  // Turn Blue LED ON
  } else {
    (GPIOE->ODR) = 0x0008;  // Turn Blue LED OFF
  }
}

// BLE service and characteristic definitions
#define DATA_CHAR_UUID "51FF12BB-3ED8-46E5-B4F9-D64E2FEC021B"
#define SERVICE_UUID   "19B10010-E8F2-537E-4F6C-D104768A1214"

BLEService dataService(SERVICE_UUID);
BLECharacteristic dataCharacteristic(DATA_CHAR_UUID, BLEWriteWithoutResponse, 20);

// Global variables for accumulating received bytes
volatile bool newValueFlag = false;  // set when start byte (0x00) is received
volatile uint8_t valueBuffer[2];     // to hold high and low byte
volatile int byteCounter = 0;          // count received bytes for current value
volatile bool receivedValueReady = false;
volatile int receivedValue = 0;        // reconstructed 16-bit integer

// LED state for received data (to toggle Blue LED)
bool blueLedState = false;

void onDataWritten(BLEDevice central, BLECharacteristic characteristic) {
  // Called when the sender writes to the characteristic
  int len = characteristic.valueLength();
  uint8_t data[len];
  characteristic.readValue(data, len);
  
  // Process each received byte individually
  for (int i = 0; i < len; i++) {
    uint8_t b = data[i];
    if (!newValueFlag) {
      // Expect a start byte: 0x00 indicates a new value is coming.
      if (b == 0x00) {
         newValueFlag = true;
         byteCounter = 0;
      }
    }
    else {
      // Accumulate the next two bytes.
      if (byteCounter < 2) {
         valueBuffer[byteCounter] = b;
         byteCounter++;
      }
      if (byteCounter == 2) {
         // Reconstruct the 16-bit value (assuming big-endian order)
         receivedValue = ((int)valueBuffer[0] << 8) | valueBuffer[1];
         receivedValueReady = true;
         newValueFlag = false;
         byteCounter = 0;
         // Toggle Blue LED to indicate reception
         blueLedState = !blueLedState;
         BlueLed(blueLedState);
       }
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("Receiver.ino starting...");
  
  // Set initial LED states: Red ON (no connection), others OFF.
  RedLed(true);
  GreenLed(false);
  BlueLed(false);
  
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }
  
  // Set the device name so the sender can find us.
  BLE.setLocalName("NextComputer");
  BLE.setAdvertisedService(dataService);
  
  // Add characteristic to the service and add service to BLE.
  dataService.addCharacteristic(dataCharacteristic);
  BLE.addService(dataService);
  
  // Register the callback to handle writes from the sender.
  dataCharacteristic.setEventHandler(BLEWritten, onDataWritten);
  
  // Start advertising.
  BLE.advertise();
  Serial.println("Bluetooth device active, waiting for connections...");
}

void loop() {
  // Poll for BLE events.
  BLEDevice central = BLE.central();
  
  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());
    
    // Update LED: connection established, so turn Red OFF and Green ON.
    RedLed(false);
    GreenLed(true);
    
    while (central.connected()) {
      // Check if a complete 16-bit value has been received.
      if (receivedValueReady) {
         // For Serial Plotter, we print only the numeric value.
         Serial.println(receivedValue);
         receivedValueReady = false;
      }
    }
    
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
    // Upon disconnection, revert to Red ON (no connection) and turn Green OFF.
    RedLed(true);
    GreenLed(false);
  }
}
