/*****************************************************************
 * GigaLED.h
 *
 * Control RGB led on Arduino Giga
 *
 * Version 1.0 Jan 2025 Leon Abelmann
 *****************************************************************/
#ifndef GigaLED_h
#define GigaLED_h

#include "Arduino.h"

class GigaLED{
 public:
  GigaLED(); // Constructor
  void Red();
  void Green();
  void Blue();
 private:
};

#endif //GigaLED_H
