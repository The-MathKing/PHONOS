#pragma once

#include <Arduino.h>
#include <AudioStream.h>
#include <arm_math.h>

/**
 * @brief FPU-Accelerated CMSIS-DSP Biquad Cascade for Formant Shifting.
 * 
 * Implements a 2-stage Direct Form I biquad cascade in float32_t.
 * Converts 16-bit Q15 audio blocks to float32_t, applies the filters,
 * and converts back to Q15 for seamless integration with Teensy Audio Library.
 */
class AudioEffectFormantBiquad : public AudioStream {
public:
    AudioEffectFormantBiquad() : AudioStream(1, inputQueueArray) {
        // Initialize CMSIS-DSP Biquad Instance for 2 stages
        arm_biquad_cascade_df1_init_f32(&biquad_inst, 2, pCoeffs, pState);
        setFormants(500.0f, 1500.0f); // Default safe values
    }

    virtual void update(void) override;

    /**
     * @brief Dynamically update F1 and F2 formant center frequencies.
     * @param f1 First formant center frequency (Hz)
     * @param f2 Second formant center frequency (Hz)
     */
    void setFormants(float f1, float f2);

private:
    audio_block_t *inputQueueArray[1];

    // CMSIS-DSP requires specific array sizing:
    // pState: 4 * numStages = 8 floats for 2 stages
    // pCoeffs: 5 * numStages = 10 floats for 2 stages
    arm_biquad_casd_df1_inst_f32 biquad_inst;
    float32_t pState[8];
    float32_t pCoeffs[10];

    // Helper to calculate BPF coefficients for a single biquad stage
    void calcBiquadBandpass(float center_freq, float Q, float32_t* coeffs_out);
};
