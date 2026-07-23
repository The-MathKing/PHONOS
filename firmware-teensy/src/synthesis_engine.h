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
     * @param exp_vector 6-DOF phonetic regression output
     * @param emo_vector 2-DOF emotion regression output
     */
    void update(const SSIExpressionVector& exp_vector, const SSIEmotionVector& emo_vector);

private:
    // Core sound generation (glottal pulse model)
    AudioSynthWaveformModulated glottal_source;
    AudioSynthWaveform          lfo_vibrato;
    
    // Emotion Dynamics
    AudioEffectWaveshaper       saturation_block;
    AudioEffectEnvelope         envelope;

    // Vocal tract formants (FPU Biquad Cascade)
    AudioEffectFormantBiquad    formant_shifter;
    
    // Output mixer (Optional, routes to I2S)
    AudioMixer4                 output_mixer;
    
    // Connections
    AudioConnection* patchCord1;
    AudioConnection* patchCord2;
    AudioConnection* patchCord3;
    AudioConnection* patchCord4;
    AudioConnection* patchCord5;

    // Pre-computed waveshaping curve for harmonic saturation
    float waveshape_curve[17];
};
