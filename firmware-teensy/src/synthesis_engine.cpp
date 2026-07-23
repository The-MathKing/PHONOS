#include "synthesis_engine.h"

// Note: In the Teensy Audio Library, objects must be instantiated globally or persistently.
// Since this class holds them as members, the SSISynthesizer instance itself must be global.

SSISynthesizer::SSISynthesizer() {
    // Route glottal source into all three formant filters in parallel
    patchCord1 = new AudioConnection(glottal_source, 0, formant1, 0);
    patchCord2 = new AudioConnection(glottal_source, 0, formant2, 0);
    patchCord3 = new AudioConnection(glottal_source, 0, formant3, 0);
    
    // Take the bandpass outputs (index 1) of the state variable filters
    patchCord4 = new AudioConnection(formant1, 1, formant_mixer, 0);
    patchCord5 = new AudioConnection(formant2, 1, formant_mixer, 1);
    patchCord6 = new AudioConnection(formant3, 1, formant_mixer, 2);
}

void SSISynthesizer::begin() {
    // Initialize audio memory (called from main.cpp)
    
    // Glottal source: Sawtooth wave mimics vocal fold vibration
    glottal_source.begin(WAVEFORM_SAWTOOTH);
    glottal_source.amplitude(0.0f); // Start muted
    glottal_source.frequency(100.0f);
    
    // Initialize formant filters (resonance Q = 5.0)
    formant1.resonance(5.0f);
    formant2.resonance(5.0f);
    formant3.resonance(5.0f);
    
    // Mix formants equally
    formant_mixer.gain(0, 0.33f);
    formant_mixer.gain(1, 0.33f);
    formant_mixer.gain(2, 0.33f);
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
    formant1.frequency(f1);
    
    // F2: 800Hz - 2500Hz
    float f2 = 800.0f + (vector.formant_2 * 1700.0f);
    formant2.frequency(f2);
    
    // F3: 2000Hz - 3500Hz
    float f3 = 2000.0f + (vector.formant_3 * 1500.0f);
    formant3.frequency(f3);
    
    // (Yaw/Timbre vector could modulate pulse width or filter resonance, 
    // left at constant for this baseline implementation).
}
