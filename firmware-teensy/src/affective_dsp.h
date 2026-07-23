/**
 * @file affective_dsp.h
 * @brief CMSIS-DSP Feature Extraction Pipeline for CNN 2: Affective Modulator
 *
 * Extracts a 4-dimensional normalized physiological feature vector from raw
 * PZT (vibration) and sEMG (muscle potential) buffers using arm_rfft_fast_f32.
 *
 * Features: [f0 (normalized), RMS (normalized), MNF (normalized), MDF (normalized)]
 *
 * Constraint: Total execution must remain under 150us on Teensy 4.1 (600MHz Cortex-M7)
 * to fit within the global 450us system clamp.
 */

#ifndef AFFECTIVE_DSP_H
#define AFFECTIVE_DSP_H

#include <arm_math.h>

// Buffer size = 256 samples at 10kHz -> 25.6ms analysis window
#define AFFECTIVE_BUFFER_SIZE 256
// FFT output = N/2 real bins (index 0..127), bin resolution = 10kHz / 256 = 39.06 Hz
#define AFFECTIVE_FFT_SIZE    256
#define AFFECTIVE_FFT_HALF    (AFFECTIVE_FFT_SIZE / 2)
#define AFFECTIVE_SAMPLE_RATE 10000.0f
#define AFFECTIVE_HZ_PER_BIN  (AFFECTIVE_SAMPLE_RATE / AFFECTIVE_FFT_SIZE)

// Normalization bounds
#define AFFECTIVE_F0_MAX      500.0f   // Hz: physiological f0 ceiling
#define AFFECTIVE_RMS_MAX     0.1f     // Volts (normalized ADC)
#define AFFECTIVE_MNF_MAX     2000.0f  // Hz: upper sEMG frequency
#define AFFECTIVE_MDF_MAX     2000.0f  // Hz

/**
 * @struct AffectiveFeatureVector
 * @brief 4-DOF normalized output vector feeding CNN 2.
 *
 * All values are in [0.0, 1.0].
 * f0          -> Vocal fundamental frequency (from PZT vibration)
 * rms_energy  -> RMS envelope amplitude (from PZT)
 * mnf         -> Mean frequency of sEMG spectrum (neuromuscular tension indicator)
 * mdf         -> Median frequency of sEMG spectrum (fatigue indicator)
 */
typedef struct {
    float f0;
    float rms_energy;
    float mnf;
    float mdf;
} AffectiveFeatureVector;

/**
 * @brief Initializes RFFT instances. Call once in setup().
 */
void affective_dsp_init();

/**
 * @brief Extracts the 4-dimensional physiological feature vector.
 *
 * @param pzt_buf    Pointer to 256-sample PZT float array  (raw normalized ADC)
 * @param semg_buf   Pointer to 256-sample sEMG float array (post-FastICA)
 * @param out        Output feature vector (f0, rms, mnf, mdf) in [0.0, 1.0]
 */
void affective_dsp_extract(const float* pzt_buf, const float* semg_buf, AffectiveFeatureVector* out);

#endif // AFFECTIVE_DSP_H
