#pragma once
#include <Arduino.h>
#include <Audio.h>
#include "tflite_micro_stub.h" // For SSIExpressionVector
#include "effect_formant_biquad.h"

/**
 * @brief SSI Synthesis Engine — Klatt-Style Formant Synthesizer
 *
 * Signal Path:
 *   LFO_Vibrato --(FM)--> GlottalSource (sawtooth)
 *       --> Saturation (harmonic distortion)
 *       --> Envelope (articulation dynamics)
 *       --> FormantBiquad (CMSIS-DSP cascaded bandpass, F1/F2)
 *       --> OutputMixer
 *       --> I2S Output (physical speaker)
 *
 * CNN Mapping:
 *   CNN1.pitch     -> glottal_source frequency (base f0)
 *   CNN1.formant_1 -> FormantBiquad F1 center frequency
 *   CNN1.formant_2 -> FormantBiquad F2 center frequency
 *   CNN1.intensity -> amplitude envelope gate
 *   CNN2.Pitch_Shift  -> f0 offset modulation
 *   CNN2.Volume_Gain  -> output_mixer master gain
 *   CNN2.arousal   -> envelope attack speed + saturation depth
 *   CNN2.valence   -> vibrato depth + formant Q tightening
 */
class SSISynthesizer {
public:
    SSISynthesizer();
    void begin();
    
    /**
     * @brief Updates synthesizer parameters continuously.
     * @param exp_vector 6-DOF phonetic regression output (CNN 1)
     * @param emo_vector 2-DOF emotion regression output (CNN 2)
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
    
    // Output mixer and I2S hardware output
    AudioMixer4                 output_mixer;
    AudioOutputI2S              i2s_output;    // Routes audio to physical speaker/DAC
    
    // Audio connections
    AudioConnection* patchCord1; // LFO -> GlottalSource FM
    AudioConnection* patchCord2; // GlottalSource -> Saturation
    AudioConnection* patchCord3; // Saturation -> Envelope
    AudioConnection* patchCord4; // Envelope -> FormantBiquad
    AudioConnection* patchCord5; // FormantBiquad -> Mixer
    AudioConnection* patchCord6; // Mixer -> I2S Left
    AudioConnection* patchCord7; // Mixer -> I2S Right (stereo copy)

    // Pre-computed waveshaping curve for harmonic saturation
    float waveshape_curve[17];
};
