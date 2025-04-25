/*****************************************************************
* GigaLED.cpp
* Control RGB led on Arduino Giga
* Version 1.0 Jan 2025 Leon Abelmann
*****************************************************************/
#include "GigaLED.h"

/*******************************************************************
 * Local functions
 ******************************************************************/

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


/*******************************************************************
 * Public functions
 ******************************************************************/

GigaLED::GigaLED(){
  // RGB led
  pinMode(86, OUTPUT);
  pinMode(87, OUTPUT);
  pinMode(88, OUTPUT);
}

void GigaLED::Red() {
  RedLed(true);
  GreenLed(false);
  BlueLed(false);
}

void GigaLED::Green() {
  RedLed(false);
  GreenLed(true);
  BlueLed(false);
}

void GigaLED::Blue() {
  RedLed(false);
  GreenLed(false);
  BlueLed(true);
}
