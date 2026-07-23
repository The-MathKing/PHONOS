/**
 * @file affective_dsp.cpp
 * @brief CMSIS-DSP Feature Extraction Pipeline — CNN 2: Affective Modulator
 *
 * Execution Budget: < 150us on Teensy 4.1 (600MHz ARM Cortex-M7 w/ hardware FPU)
 *
 * Pipeline:
 *   1. PZT Buffer  -> arm_rfft_fast_f32 -> Autocorrelation peak -> f0 (Hz)
 *                  -> arm_rms_f32       -> RMS energy envelope
 *   2. sEMG Buffer -> arm_rfft_fast_f32 -> Power spectrum -> MNF, MDF
 *
 * All outputs are normalized to [0.0, 1.0] for direct CNN 2 tensor injection.
 */

#include "affective_dsp.h"
#include <Arduino.h>  // for constrain()
#include <arm_math.h>

// Static RFFT instances — statically allocated to avoid runtime heap use
static arm_rfft_fast_instance_f32 pzt_rfft_instance;
static arm_rfft_fast_instance_f32 semg_rfft_instance;

// Working scratch buffers — statically allocated in DTCM RAM
static float pzt_fft_out[AFFECTIVE_FFT_SIZE];
static float semg_fft_out[AFFECTIVE_FFT_SIZE];
static float pzt_mag[AFFECTIVE_FFT_HALF];
static float semg_power[AFFECTIVE_FFT_HALF];

// ─────────────────────────────────────────────────────────────────────────────

void affective_dsp_init() {
    // arm_rfft_fast_init_f32 is lightweight and must be called once at startup
    arm_rfft_fast_init_f32(&pzt_rfft_instance,  AFFECTIVE_FFT_SIZE);
    arm_rfft_fast_init_f32(&semg_rfft_instance, AFFECTIVE_FFT_SIZE);
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Estimates the Fundamental Frequency (f0) from a PZT magnitude spectrum.
 *
 * Uses the Harmonic Product Spectrum (HPS) method: multiply the magnitude
 * spectrum downsampled by factors 1, 2, and 3. The peak of the product is f0.
 * This is computationally minimal and robust to glottal source harmonics.
 */
static float extract_f0(const float* mag, uint32_t half_n) {
    float max_val = 0.0f;
    uint32_t max_idx = 1; // Skip DC bin at index 0

    // HPS product across first 1/3 of spectrum to stay in vocal range (80-500Hz)
    const uint32_t search_limit = half_n / 3;

    for (uint32_t i = 1; i < search_limit; i++) {
        // Harmonic product: index i, 2i, 3i
        uint32_t i2 = i * 2;
        uint32_t i3 = i * 3;
        if (i3 >= half_n) break;

        float product = mag[i] * mag[i2] * mag[i3];
        if (product > max_val) {
            max_val = product;
            max_idx = i;
        }
    }

    return (float)max_idx * AFFECTIVE_HZ_PER_BIN;
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Computes Mean Frequency (MNF) and Median Frequency (MDF) from power spectrum.
 *
 * MNF = sum(f[i] * P[i]) / sum(P[i])    — centroid of the sEMG power spectrum
 * MDF = frequency at which cumulative power reaches 50% of total — muscle fatigue indicator
 */
static void extract_mnf_mdf(const float* power, uint32_t half_n, float* mnf_out, float* mdf_out) {
    float total_power = 0.0f;
    float weighted_sum = 0.0f;

    // Single pass for total power and weighted sum
    for (uint32_t i = 1; i < half_n; i++) {
        float freq = (float)i * AFFECTIVE_HZ_PER_BIN;
        total_power  += power[i];
        weighted_sum += freq * power[i];
    }

    // MNF
    *mnf_out = (total_power > 1e-9f) ? (weighted_sum / total_power) : 0.0f;

    // MDF — scan for 50% cumulative power threshold
    float cumulative = 0.0f;
    const float half_power = total_power * 0.5f;
    *mdf_out = 0.0f;
    for (uint32_t i = 1; i < half_n; i++) {
        cumulative += power[i];
        if (cumulative >= half_power) {
            *mdf_out = (float)i * AFFECTIVE_HZ_PER_BIN;
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

void affective_dsp_extract(const float* pzt_buf, const float* semg_buf, AffectiveFeatureVector* out) {
    // ── PZT: f0 and RMS Energy ────────────────────────────────────────────────

    // 1. RMS energy of raw PZT buffer using CMSIS-DSP hardware intrinsic
    float rms_raw = 0.0f;
    arm_rms_f32(const_cast<float*>(pzt_buf), AFFECTIVE_BUFFER_SIZE, &rms_raw);

    // 2. RFFT of PZT buffer -> complex interleaved output
    // arm_rfft_fast_f32 returns [DC_real, Nyquist_real, r1, i1, r2, i2, ...]
    arm_rfft_fast_f32(&pzt_rfft_instance, const_cast<float*>(pzt_buf), pzt_fft_out, 0);

    // 3. Compute magnitude spectrum (skip DC [0] and Nyquist [1])
    // Output pairs starting at index 2: (real, imag) for bins 1..N/2-1
    arm_cmplx_mag_f32(&pzt_fft_out[2], &pzt_mag[1], AFFECTIVE_FFT_HALF - 1);
    pzt_mag[0] = pzt_fft_out[0]; // DC

    // 4. Extract f0 using Harmonic Product Spectrum
    float f0 = extract_f0(pzt_mag, AFFECTIVE_FFT_HALF);

    // ── sEMG: MNF and MDF ────────────────────────────────────────────────────

    // 5. RFFT of sEMG buffer
    arm_rfft_fast_f32(&semg_rfft_instance, const_cast<float*>(semg_buf), semg_fft_out, 0);

    // 6. Compute power spectrum (|X[f]|^2) from complex FFT output
    arm_cmplx_mag_squared_f32(&semg_fft_out[2], &semg_power[1], AFFECTIVE_FFT_HALF - 1);
    semg_power[0] = semg_fft_out[0] * semg_fft_out[0]; // DC power

    // 7. Extract MNF and MDF
    float mnf = 0.0f;
    float mdf = 0.0f;
    extract_mnf_mdf(semg_power, AFFECTIVE_FFT_HALF, &mnf, &mdf);

    // ── Normalize to [0.0, 1.0] for CNN 2 tensor input ──────────────────────
    out->f0         = fmaxf(0.0f, fminf(1.0f, f0        / AFFECTIVE_F0_MAX));
    out->rms_energy = fmaxf(0.0f, fminf(1.0f, rms_raw   / AFFECTIVE_RMS_MAX));
    out->mnf        = fmaxf(0.0f, fminf(1.0f, mnf       / AFFECTIVE_MNF_MAX));
    out->mdf        = fmaxf(0.0f, fminf(1.0f, mdf       / AFFECTIVE_MDF_MAX));
}
