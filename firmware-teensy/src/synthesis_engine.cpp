#include "synthesis_engine.h"

// Note: In the Teensy Audio Library, objects must be instantiated globally or persistently.
// Since this class holds them as members, the SSISynthesizer instance itself must be global.

SSISynthesizer::SSISynthesizer() {
    // FM modulation for vibrato
    patchCord1 = new AudioConnection(lfo_vibrato, 0, glottal_source, 0);
    // Audio path
    patchCord2 = new AudioConnection(glottal_source, 0, saturation_block, 0);
    patchCord3 = new AudioConnection(saturation_block, 0, envelope, 0);
    patchCord4 = new AudioConnection(envelope, 0, formant_shifter, 0);
    patchCord5 = new AudioConnection(formant_shifter, 0, output_mixer, 0);
}

void SSISynthesizer::begin() {
    // Initialize audio memory (called from main.cpp)
    
    // Glottal source: Sawtooth wave mimics vocal fold vibration
    glottal_source.begin(WAVEFORM_SAWTOOTH);
    glottal_source.amplitude(0.0f); // Start muted
    glottal_source.frequency(100.0f);
    
    // LFO (Vibrato)
    lfo_vibrato.begin(WAVEFORM_SINE);
    lfo_vibrato.amplitude(0.0f);
    lfo_vibrato.frequency(5.0f);

    // Identity Waveshape curve initialization
    for (int i=0; i<17; i++) {
        waveshape_curve[i] = (i - 8.0f) / 8.0f; 
    }
    saturation_block.shape(waveshape_curve, 17);

    // Envelope
    envelope.attack(10.0f);
    envelope.decay(10.0f);
    envelope.sustain(1.0f);
    envelope.release(10.0f);
    
    // Mixers
    output_mixer.gain(0, 1.0f);
}

void SSISynthesizer::update(const SSIExpressionVector& exp_vector, const SSIEmotionVector& emo_vector) {
    // 1. Emotion: Low Arousal / Negative Valence (Pitch Downshift & Vibrato)
    float base_pitch_offset = 0.0f;
    float lfo_amp = 0.0f;
    float q_factor = 5.0f;
    
    if (emo_vector.arousal < 0.3f || emo_vector.valence < -0.3f) {
        base_pitch_offset = -20.0f; // Downshift pitch
        lfo_amp = 0.05f; // Add vibrato depth
        q_factor = 10.0f; // Tighter formants
    }
    
    lfo_vibrato.amplitude(lfo_amp);
    formant_shifter.setQ(q_factor);

    // 2. Emotion: High Arousal (Fast Attack & Saturation)
    if (emo_vector.arousal > 0.7f) {
        envelope.attack(2.0f);
        // Simple saturation mapping logic (steepness)
        float steepness = 1.0f + (emo_vector.arousal - 0.7f) * 5.0f;
        for (int i=0; i<17; i++) {
            float x = (i - 8.0f) / 8.0f;
            waveshape_curve[i] = constrain(x * steepness, -1.0f, 1.0f);
        }
        saturation_block.shape(waveshape_curve, 17);
    } else {
        envelope.attack(15.0f); // Slower attack
        for (int i=0; i<17; i++) {
            waveshape_curve[i] = (i - 8.0f) / 8.0f; 
        }
        saturation_block.shape(waveshape_curve, 17);
    }

    // 3. Phonetic Pitch mapping
    float f0 = 80.0f + (exp_vector.pitch * 220.0f) + base_pitch_offset;
    glottal_source.frequency(f0);
    
    // 4. Phonetic Intensity mapping
    float amp = constrain(exp_vector.intensity, 0.0f, 1.0f);
    glottal_source.amplitude(amp);
    if (amp > 0.1f) envelope.noteOn();
    else envelope.noteOff();
    
    // 5. Phonetic Formant mapping
    float f1 = 300.0f + (exp_vector.formant_1 * 700.0f);
    float f2 = 800.0f + (exp_vector.formant_2 * 1700.0f);
    
    formant_shifter.setFormants(f1, f2);
}
