#include "effect_formant_biquad.h"

void AudioEffectFormantBiquad::update(void) {
    audio_block_t *block = receiveWritable(0);
    if (!block) return;

    // CMSIS-DSP FPU conversion: Q15 (int16_t) to float32_t
    float32_t float_buffer[AUDIO_BLOCK_SAMPLES];
    arm_q15_to_float(block->data, float_buffer, AUDIO_BLOCK_SAMPLES);

    // Hardware FPU Biquad Cascade execution (2 stages)
    // Note: In-place processing is supported by CMSIS-DSP biquad df1
    arm_biquad_cascade_df1_f32(&biquad_inst, float_buffer, float_buffer, AUDIO_BLOCK_SAMPLES);

    // Convert float32_t back to Q15 (int16_t) and store in original block
    arm_float_to_q15(float_buffer, block->data, AUDIO_BLOCK_SAMPLES);

    transmit(block);
    release(block);
}

void AudioEffectFormantBiquad::calcBiquadBandpass(float center_freq, float Q, float32_t* coeffs_out) {
    // Standard RBJ Audio EQ Cookbook equations for Bandpass Filter (0dB peak)
    float omega = 2.0f * (float)PI * center_freq / AUDIO_SAMPLE_RATE_EXACT;
    float alpha = sinf(omega) / (2.0f * Q);

    float b0 = alpha;
    float b1 = 0.0f;
    float b2 = -alpha;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cosf(omega);
    float a2 = 1.0f - alpha;

    // CMSIS-DSP arm_biquad_cascade_df1_f32 expects normalized coefficients 
    // in the exact order: {b0, b1, b2, a1, a2}. 
    // BUT note that CMSIS-DSP actually expects a1 and a2 to be NEGATED compared 
    // to standard difference equations! (y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] + a1*y[n-1] + a2*y[n-2])
    // Standard EQ uses -a1 and -a2 for the recursive part.
    coeffs_out[0] = b0 / a0;
    coeffs_out[1] = b1 / a0;
    coeffs_out[2] = b2 / a0;
    coeffs_out[3] = -a1 / a0;
    coeffs_out[4] = -a2 / a0;
}

void AudioEffectFormantBiquad::setFormants(float f1, float f2) {
    // Fixed Q-factor for vocal tract resonance simulation
    float Q = 5.0f; 

    // Stage 1: F1 Resonance
    calcBiquadBandpass(f1, Q, &pCoeffs[0]);

    // Stage 2: F2 Resonance
    calcBiquadBandpass(f2, Q, &pCoeffs[5]);
}
