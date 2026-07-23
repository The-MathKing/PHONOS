#include "synthesis_engine.h"

// Note: In the Teensy Audio Library, objects must be instantiated globally or persistently.
// Since this class holds them as members, the SSISynthesizer instance itself must be global.

SSISynthesizer::SSISynthesizer() {
    // Vibrato LFO -> Glottal Source FM input
    patchCord1 = new AudioConnection(lfo_vibrato, 0, glottal_source, 0);
    // Main audio signal path
    patchCord2 = new AudioConnection(glottal_source, 0, saturation_block, 0);
    patchCord3 = new AudioConnection(saturation_block, 0, envelope, 0);
    patchCord4 = new AudioConnection(envelope, 0, formant_shifter, 0);
    patchCord5 = new AudioConnection(formant_shifter, 0, output_mixer, 0);
    // I2S hardware output — routes physical audio to speaker/DAC
    patchCord6 = new AudioConnection(output_mixer, 0, i2s_output, 0); // Left
    patchCord7 = new AudioConnection(output_mixer, 0, i2s_output, 1); // Right (mono copy)
}

void SSISynthesizer::begin() {
    // Glottal source: Sawtooth wave mimics vocal fold vibration
    glottal_source.begin(WAVEFORM_SAWTOOTH);
    glottal_source.amplitude(0.0f); // Start muted until first inference
    glottal_source.frequency(120.0f); // Default male vocal f0

    // Vibrato LFO
    lfo_vibrato.begin(WAVEFORM_SINE);
    lfo_vibrato.amplitude(0.0f);
    lfo_vibrato.frequency(5.0f); // 5Hz natural vibrato rate

    // Identity waveshape curve (linear = no saturation at rest)
    for (int i = 0; i < 17; i++) {
        waveshape_curve[i] = (i - 8.0f) / 8.0f;
    }
    saturation_block.shape(waveshape_curve, 17);

    // Envelope (ADSR): voiced phoneme attack/release
    envelope.attack(10.0f);
    envelope.decay(10.0f);
    envelope.sustain(1.0f);
    envelope.release(30.0f);

    // Output mixer: start at unity gain
    output_mixer.gain(0, 1.0f);
}

void SSISynthesizer::update(const SSIExpressionVector& exp_vector, const SSIEmotionVector& emo_vector) {
    // ─── CNN 2: Affective Prosodic Modulation ────────────────────────────────
    // Pitch_Shift: CNN2 output[0] -> f0 transposition in Hz
    // Range: -50Hz (low arousal) to +80Hz (high arousal)
    float pitch_shift_hz = (emo_vector.arousal - 0.5f) * 130.0f;

    // Volume_Gain: CNN2 output[1] -> master output mixer amplitude [0.2, 1.0]
    float volume_gain = 0.2f + (emo_vector.arousal * 0.8f);
    output_mixer.gain(0, constrain(volume_gain, 0.0f, 1.0f));

    // Vibrato depth driven by valence: negative valence increases vibrato
    float lfo_amp = 0.0f;
    float q_factor = 5.0f;
    if (emo_vector.valence < -0.2f) {
        lfo_amp = fabsf(emo_vector.valence) * 0.08f;
        q_factor = 10.0f; // Tighter, more nasal formants
    }
    lfo_vibrato.amplitude(lfo_amp);
    formant_shifter.setQ(q_factor);

    // ─── CNN 2: High Arousal — Fast Attack + Saturation Depth ───────────────
    if (emo_vector.arousal > 0.7f) {
        envelope.attack(2.0f);
        float steepness = 1.0f + (emo_vector.arousal - 0.7f) * 5.0f;
        for (int i = 0; i < 17; i++) {
            float x = (i - 8.0f) / 8.0f;
            waveshape_curve[i] = constrain(x * steepness, -1.0f, 1.0f);
        }
        saturation_block.shape(waveshape_curve, 17);
    } else {
        envelope.attack(15.0f);
        for (int i = 0; i < 17; i++) {
            waveshape_curve[i] = (i - 8.0f) / 8.0f; // Linear (no saturation)
        }
        saturation_block.shape(waveshape_curve, 17);
    }

    // ─── CNN 1: Phonetic Pitch Mapping ───────────────────────────────────────
    // exp_vector.pitch in [0.0, 1.0] -> glottal f0 in [80, 300] Hz
    // CNN 2 Pitch_Shift modulates on top for prosodic expression
    float f0_base = 80.0f + (exp_vector.pitch * 220.0f);
    float f0_final = constrain(f0_base + pitch_shift_hz, 60.0f, 400.0f);
    glottal_source.frequency(f0_final);

    // ─── CNN 1: Phonetic Intensity (Voicing Gate) ────────────────────────────
    float amp = constrain(exp_vector.intensity, 0.0f, 1.0f);
    glottal_source.amplitude(amp);
    if (amp > 0.05f) envelope.noteOn();
    else             envelope.noteOff();

    // ─── CNN 1: Formant Mapping (Klatt-Style Vocal Tract) ───────────────────
    // F1: first formant [300, 1000] Hz — jaw height / vowel openness
    // F2: second formant [800, 2500] Hz — tongue front-back position
    float f1 = 300.0f + (exp_vector.formant_1 * 700.0f);
    float f2 = 800.0f + (exp_vector.formant_2 * 1700.0f);
    formant_shifter.setFormants(f1, f2);
}
