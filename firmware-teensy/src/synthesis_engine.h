#pragma once
#include <Arduino.h>
#include <Audio.h>
#include "tflite_micro_stub.h" // For SSIExpressionVector

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
    
    // Vocal tract formants (State Variable Filters)
    AudioFilterStateVariable formant1;
    AudioFilterStateVariable formant2;
    AudioFilterStateVariable formant3;
    
    // Mixers
    AudioMixer4              formant_mixer;
    AudioMixer4              output_mixer;
    
    // Connections
    AudioConnection* patchCord1;
    AudioConnection* patchCord2;
    AudioConnection* patchCord3;
    AudioConnection* patchCord4;
    AudioConnection* patchCord5;
    AudioConnection* patchCord6;
};
