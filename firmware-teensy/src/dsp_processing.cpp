#include "dsp_processing.h"

// CMSIS-DSP Biquad Instances for 20Hz High-Pass Filter (A0 and A1)
static arm_biquad_casd_df1_inst_f32 hpf_inst_a0;
static arm_biquad_casd_df1_inst_f32 hpf_inst_a1;
static float32_t hpf_state_a0[4];
static float32_t hpf_state_a1[4];

// Pre-calculated coefficients for 2nd-order Butterworth HPF (Fc=20Hz, Fs=2000Hz)
// CMSIS-DSP expects: b0, b1, b2, a1, a2 (where a1 and a2 are inverted in sign)
static const float32_t hpf_coeffs[5] = {
    0.95654323f, -1.91308645f, 0.95654323f, 
    1.91119707f, -0.91497583f // a1, a2
};

void dsp_fastica_init() {
    arm_biquad_cascade_df1_init_f32(&hpf_inst_a0, 1, (float32_t*)hpf_coeffs, hpf_state_a0);
    arm_biquad_cascade_df1_init_f32(&hpf_inst_a1, 1, (float32_t*)hpf_coeffs, hpf_state_a1);
}

void dsp_fastica_process(
    const uint16_t raw_in[4][512],
    float          sources[4][512],
    uint8_t        num_ch,
    uint16_t       num_samp
) {
    // 1. Convert to float and mean-center
    float f_a0[512];
    float f_a1[512];
    float mean_a0 = 0, mean_a1 = 0;
    
    for (int i = 0; i < 512; i++) {
        f_a0[i] = (float)raw_in[0][i];
        f_a1[i] = (float)raw_in[1][i];
        sources[2][i] = (float)raw_in[2][i]; // PZT passthrough
        sources[3][i] = (float)raw_in[3][i]; // EDA passthrough
    }
    
    arm_mean_f32(f_a0, 512, &mean_a0);
    arm_mean_f32(f_a1, 512, &mean_a1);
    
    arm_offset_f32(f_a0, -mean_a0, f_a0, 512);
    arm_offset_f32(f_a1, -mean_a1, f_a1, 512);

    // 2. High-Pass Filter (20 Hz Butterworth) to remove baseline drift
    arm_biquad_cascade_df1_f32(&hpf_inst_a0, f_a0, f_a0, 512);
    arm_biquad_cascade_df1_f32(&hpf_inst_a1, f_a1, f_a1, 512);

    // 3. FastICA 2x2 Matrix Multiplication Loop
    // Simplified stub matrix mapping for performance testing 
    // In production, W is iteratively estimated via negentropy max
    float W[2][2] = {
        { 0.8f, -0.2f },
        {-0.2f,  0.8f }
    };

    for (int i = 0; i < 512; i++) {
        sources[0][i] = (W[0][0] * f_a0[i]) + (W[0][1] * f_a1[i]);
        sources[1][i] = (W[1][0] * f_a0[i]) + (W[1][1] * f_a1[i]);
    }
}
