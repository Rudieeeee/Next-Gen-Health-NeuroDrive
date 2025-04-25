/**************************************************************
 * Wheelchair.ino
 *
 * Leon Abelmann, January 2025
 *
 * Version 1.1 - Tranferred RGB led control to GigaLED class
 *
 * Version 1.0 - Single button and bluetooth, digital control of R-NET
 * operated electric wheelchair through an IOM module.
 *
 * On first button press, port 1 of output DIN will go up: IOM forward
 * On next button press, port 2 goes up: IOM Backward etc Instead of
 * button, Giga also monitors bluetooth port. If a 1 is received, same
 * procedure (forward/backward) is followed
 **************************************************************/
#include "Wheelchair.h"

// Constants
// Pins definition for Arduino Giga
const int buttonPin = D2;       // pin connected to pushbottom (which pulls to gnd)
const int IOM1      = D3;       // DAC output to pin 1 of 9-pin DIN plug
const int IOM2      = D4;       // DAC output to pin 2-4 of 9-pin DIN plug

// Global variables
// Debounce button:
int buttonState;                // the current reading from the input pin
int lastButtonState = LOW;      // the previous reading from the input pin
// the following variables are unsigned longs because the time, measured in
// milliseconds, will quickly become a bigger number than can be stored in an int.
unsigned long lastDebounceTime = 0;  // the last time the output pin was toggled
unsigned long debounceDelay = 50;    // the debounce time; increase if the output flickers

// LED indicator
int ledState = HIGH;            // current state of led output pin
GigaLED LED;

// Bluetooth
BLEService ledService("0001"); // Bluetooth Low Energy LED Service

// Bluetooth Low Energy LED Switch Characteristic
// - custom 128-bit UUID, read and writable by central
BLEByteCharacteristic switchCharacteristic("0001", BLEWrite);


/**************************************************************
 * Setup()
 *
 * Sets pin states to in- or output Set global variables Initialized
 * serial communication to host computer (if connected)
 *************************************************************/
void setup() {

  // Set pin modes for button and communication with IOM
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(IOM1, OUTPUT);
  analogWrite(IOM1,255);
  pinMode(IOM2, OUTPUT);  
  analogWrite(IOM2,255);

  // Serial communication (for debugging)
  Serial.begin(115200);
  delay(1000);
  Serial.write("Wheelchair.ino\n");

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


/**************************************************************
 * ReadButton()
 * 
 * Detects whether button is pressed. Run inside main loop.
 * Returns true if button press is detected
 * 
 * https://www.arduino.cc/en/Tutorial/BuiltInExamples/Debounce
 *************************************************************/
bool readButton() {
  // read the state of the switch into a local variable:
  int reading = digitalRead(buttonPin);
  // Serial.print("Button : ");
  // Serial.println(reading);
  bool buttonPressed = false;

  // check to see if you just pressed the button
  // (i.e. the input went from LOW to HIGH), and you've waited long enough
  // since the last press to ignore any noise:

  // If the switch changed, due to noise or pressing:
  if (reading != lastButtonState) {
    // reset the debouncing timer
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    // whatever the reading is at, it's been there for longer than the debounce
    // delay, so take it as the actual current state:

    // if the button state has changed:
    if (reading != buttonState) {
      buttonState = reading;

      // only toggle the LED if the new button state is HIGH
      if (buttonState == HIGH) {
 	buttonPressed = true;
      }
    }
  }

  // save the reading. Next time through the loop, it'll be the lastButtonState:
  lastButtonState = reading;
  return buttonPressed;
}

/**************************************************************
* toggleDirection()
* Change the direction of the wheelchair, and indicate with green led
* Set either IOM1 (forward) or IOM2 (reverse) high for 1 second
*************************************************************/
void toggleDirection() {
  // Toggle led
  ledState = !ledState;

  // set the LED:
  if (ledState) {
    LED.Blue();
  }
  else {
    LED.Red();
  }

  // Send debug info to computer
  Serial.write("Button Pressed: ");
  if (ledState) {
    Serial.write("on\n");
    // Set pin IOM1 high for one second
    analogWrite(IOM1,0);
    delay(1000);
    analogWrite(IOM1,255);
  }
  else {
    Serial.write("off\n");
    // Set pin IOM2 high for one second
    analogWrite(IOM2,0);
    delay(1000);
    analogWrite(IOM2,255);
  }
}

void loop() {
  // Even without Bluetooth, check if button is pressed, activate wheelchair
  if (readButton()) {
    toggleDirection();
  }    

  // Without bluetooth, LED is red
  LED.Red();
  // Check bluetooth
  BLEDevice central = BLE.central();

  // if a central is connected to peripheral:
  if (central) {
    Serial.print("Connected to central: ");
    LED.Blue();
    // print the central's MAC address:
    Serial.println(central.address());

    // while the central is still connected to peripheral:
    while (central.connected()) {
      LED.Green();
      // if the remote device wrote to the characteristic,
      // use the value to control the LED:
      if (switchCharacteristic.written()) {
	Serial.write("BT command: ");
	Serial.print((int)switchCharacteristic.value());
	Serial.write(", ");
        if (switchCharacteristic.value() == 1) {
	  toggleDirection();
        }
	else { // switchCharacteristic <> 1
	  Serial.write(" ignore\n");
	}
      }
      // Also react on buttonpressed:
      if (readButton()) {
	toggleDirection();
      }
    }    
    Serial.print(F("Disconnected from central: "));
    Serial.println(central.address());
  }
}
