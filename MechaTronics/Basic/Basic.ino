/**************************************************************
* Basic.ino
* Version 1.0 - Single button, digital control of R-NET operated
* electric wheelchair through an IOM module.
* Leon Abelmann, August 2024
* On button press, port 1 of output DIN will go up: IOM forward
* Pin 2 (reverse), Pin 3 (left) and Pin 4 (right) are kept high
**************************************************************/

// Constants
// Pins definition for Arduino Giga
const int buttonPin = D2;       // pin connected to pushbottom (which pulls to gnd)
const int ledPin    = PIN_LED;  // LED pin (onboard)
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
 
/**************************************************************
void Setup()
Sets pin states to in- or output
Set global variables
Initialized serial communication to host computer (if connected)
*************************************************************/
void setup() {
  // Set pin modes and states
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, ledState);
  pinMode(IOM1, OUTPUT);
  analogWrite(IOM1,255);
  pinMode(IOM2, OUTPUT);  
  analogWrite(IOM2,255);

  // Serial communication (for debugging)
  Serial.begin(115200);
  delay(1000);
  Serial.write("Basic.ino\n");
}

/**************************************************************
bool ReadButton()
Detects whether button is pressed. Run inside main loop.
Returns true if button press is detected
https://www.arduino.cc/en/Tutorial/BuiltInExamples/Debounce
*************************************************************/
bool readButton() {
  // read the state of the switch into a local variable:
  int reading = digitalRead(buttonPin);
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

void loop() {
  // Check if button is pressed
  if (readButton()) {
    // Toggle led
    ledState = !ledState;
    // set the LED:
    digitalWrite(ledPin, ledState);
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
  
  // Slow down loop
  delay(10);
}
