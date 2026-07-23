#pragma once
#include <Arduino.h>
#include <Audio.h>
#include "tflite_micro_stub.h" // For SSIExpressionVector
#include "effect_formant_biquad.h"

/**
 * @brief SSI Synthesis Engine
 * Maps 6-DOF regression output to real-time audio parameters.
 */
class SSISynthesizer {
public:
    SSISynthesizer();
    void begin();
    
    /**
     * @brief Updates synthesizer parameters continuously.
     * @param vector 6-DOF regression output from the CNN
     */
    void update(const SSIExpressionVector& vector);

private:
    // Core sound generation (glottal pulse model)
    AudioSynthWaveform       glottal_source;
    
    // Vocal tract formants (FPU Biquad Cascade)
    AudioEffectFormantBiquad formant_shifter;
    
    // Output mixer (Optional, routes to I2S)
    AudioMixer4              output_mixer;
    
    // Connections
    AudioConnection* patchCord1;
    AudioConnection* patchCord2;
};
