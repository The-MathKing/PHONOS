#pragma once
#include <Arduino.h>

/**
 * @brief Streams raw ADC values in a comma-separated format compatible 
 *        with the Arduino IDE Serial Plotter.
 *        Used for Week 8 micro-volt validation and active shielding tests.
 * 
 * @param ch0 L-sEMG ADC value
 * @param ch1 R-sEMG ADC value
 * @param ch2 MMG (PZT) ADC value
 */
inline void stream_telemetry_plotter(uint16_t ch0, uint16_t ch1, uint16_t ch2) {
    Serial.print("L-sEMG:");
    Serial.print(ch0);
    Serial.print(",");
    
    Serial.print("R-sEMG:");
    Serial.print(ch1);
    Serial.print(",");
    
    Serial.print("MMG:");
    Serial.println(ch2);
}
