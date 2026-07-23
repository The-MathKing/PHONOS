#include "synthesis_engine.h"

// Note: In the Teensy Audio Library, objects must be instantiated globally or persistently.
// Since this class holds them as members, the SSISynthesizer instance itself must be global.

SSISynthesizer::SSISynthesizer() {
    // Route glottal source into formant shifter
    patchCord1 = new AudioConnection(glottal_source, 0, formant_shifter, 0);
    // Route formant shifter out to mixer
    patchCord2 = new AudioConnection(formant_shifter, 0, output_mixer, 0);
}

void SSISynthesizer::begin() {
    // Initialize audio memory (called from main.cpp)
    
    // Glottal source: Sawtooth wave mimics vocal fold vibration
    glottal_source.begin(WAVEFORM_SAWTOOTH);
    glottal_source.amplitude(0.0f); // Start muted
    glottal_source.frequency(100.0f);
    
    // Mixers
    output_mixer.gain(0, 1.0f);
}

void SSISynthesizer::update(const SSIExpressionVector& vector) {
    // 1. Pitch mapping: Map [0, 1] to human fundamental frequency range [80Hz, 300Hz]
    float f0 = 80.0f + (vector.pitch * 220.0f);
    glottal_source.frequency(f0);
    
    // 2. Intensity mapping: Map [0, 1] to amplitude
    float amp = constrain(vector.intensity, 0.0f, 1.0f);
    glottal_source.amplitude(amp);
    
    // 3. Formant mapping: Map to typical vowel formant frequency ranges
    // F1: 300Hz - 1000Hz
    float f1 = 300.0f + (vector.formant_1 * 700.0f);
    
    // F2: 800Hz - 2500Hz
    float f2 = 800.0f + (vector.formant_2 * 1700.0f);
    
    // Dynamically update the hardware FPU biquad cascade
    formant_shifter.setFormants(f1, f2);
}
