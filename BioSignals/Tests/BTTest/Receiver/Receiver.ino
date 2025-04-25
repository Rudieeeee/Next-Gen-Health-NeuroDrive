/**************************************************************
* Receiver.ino
* Bluetooth peripheral device (wheelchair).
* When something is send to the port, the led switches on/off
**************************************************************/
// Bluetooth
#include <ArduinoBLE.h>

// Global variables

// LED indicator
int ledState = HIGH;            // current state of led output pin

// Bluetooth
BLEService ledService("0001"); // Bluetooth Low Energy LED Service

// Bluetooth Low Energy LED Switch Characteristic
// - custom 128-bit UUID, read and writable by central
BLEByteCharacteristic switchCharacteristic("0001", BLEWrite);


/**************************************************************
void Setup()
Sets pin states to in- or output
Set global variables
Initialized serial communication to host computer (if connected)
*************************************************************/
void setup() {
  // Set pin modes for RGB led
  pinMode(86, OUTPUT);
  pinMode(87, OUTPUT);
  pinMode(88, OUTPUT);

  // Serial communication (for debugging)
  Serial.begin(115200);
  delay(1000);
  Serial.write("Receiver.ino\n");

  // Initialize bluetooth
  if (!BLE.begin()) {
    Serial.println("starting Bluetooth Low Energy module failed!");
    while (1);
  }

  // Print bluetooth address of this device
  String address = BLE.address();

  Serial.print("Local address is: ");
  Serial.println(address);
  
  // set advertised local name and service UUID:
  BLE.setDeviceName("Wheelchair");
  BLE.setLocalName("Wheelchair");
  BLE.advertise();
  BLE.setAdvertisedService(ledService);

  // add the characteristic to the service
  ledService.addCharacteristic(switchCharacteristic);

  // add service
  BLE.addService(ledService);

  // set the initial value for the characeristic:
  switchCharacteristic.writeValue(0);

  // start advertising
  BLE.advertise();

  Serial.println("Bluetooth ready");
}

void RedLed(bool state){
  if (state) {
    (GPIOI->ODR) = 0x0000;
  }
  else {
    (GPIOI->ODR) = 0x1000;
  }
}

void GreenLed(bool state){
  if (state) {
    (GPIOJ->ODR) = 0x0000;
  }
  else {
    (GPIOJ->ODR) = 0x2000;
  }
}

void BlueLed(bool state){
  if (state) {
    (GPIOE->ODR) = 0x0000;
  }
  else {
    (GPIOE->ODR) = 0x0008;
  }
}

void loop() {
  // Without bluetooth, LED is red
  BlueLed(false);
  GreenLed(false);
  RedLed(true);
  // Check bluetooth
  BLEDevice central = BLE.central();

  // if a central is connected to peripheral:
  if (central) {
    Serial.print("Connected to central: ");
    RedLed(false);
    BlueLed(true);
    // print the central's MAC address:
    Serial.println(central.address());

    // while the central is still connected to peripheral:
    while (central.connected()) {
      BlueLed(false);
      // if the remote device wrote to the characteristic,
      // use the value to control the LED:
      if (switchCharacteristic.written()) {
	Serial.write("BT command: ");
	Serial.print((int)switchCharacteristic.value());
	Serial.write(", ");
        if (switchCharacteristic.value() == 1) { 
	  // Toggle led
	  ledState = !ledState;
	  // set the LED:
	  GreenLed(ledState);
	  if (ledState) {
	    Serial.write("on\n");
	  }
	  else {
	    Serial.write("off\n");
	  }
        }
	else { // switchCharacteristic <> 1
	  Serial.write(" ignore\n");
	}
      }
    }
    // when the central disconnects, print it out:
    Serial.print(F("Disconnected from central: "));
    Serial.println(central.address());
  }
}
