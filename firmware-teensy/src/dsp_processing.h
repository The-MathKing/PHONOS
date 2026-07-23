/**
 * @file    dsp_processing.h
 * @brief   SSI DSP Layer — Fixed FastICA Source Separation (float32 / Cortex-M7 FPU)
 * @target  Teensy 4.1 (ARM Cortex-M7, VFPv5-D16 hard-float ABI)
 *
 * Provides a 3-channel Fast Independent Component Analysis (FastICA) pipeline
 * that takes raw multi-channel biosignal data and isolates independent
 * source components. The primary goal is to separate:
 *
 *   S0 — True subvocal motor intent (laryngeal + submental muscle activity)
 *   S1 — Macroscopic jaw / swallowing motion artifact
 *   S2 — MMG PZT vibration (mechanomyography tissue resonance)
 *
 * Algorithm: Negentropy-maximization FastICA using a hyperbolic tangent (tanh)
 * nonlinearity. Runs fully in float32 leveraging the M7 FPU registers.
 *
 * Mathematical Foundation:
 * ─────────────────────────
 * Given observation matrix X (channels × samples), the ICA model assumes:
 *
 *     X = A · S    where A is the unknown mixing matrix, S are source signals
 *
 * FastICA estimates the demixing matrix W = A⁻¹ by iterating:
 *
 *     w ← (1/N) Σ [ x · g(wᵀx) ] − E[g′(wᵀx)] · w
 *     w ← w / ‖w‖
 *
 * where g(u) = tanh(α·u)  (negentropy contrast function, α ≈ 1.0)
 *       g′(u) = α · (1 − tanh²(α·u))
 *
 * Preprocessing:
 *     1. Mean centering:  X̃ = X − μ
 *     2. Whitening:       Z = W_white · X̃  such that E[ZZᵀ] = I
 *        (via eigendecomposition of covariance matrix: Σ = EΛEᵀ → W_white = Λ^(-½)Eᵀ)
 *
 * References:
 *     Hyvärinen & Oja (2000). "Independent Component Analysis: Algorithms and
 *     Applications." Neural Networks, 13(4-5), pp. 411-430.
 */

#pragma once

#include <Arduino.h>
#include <stdint.h>

// ── Configuration Constants ──────────────────────────────────────────────────
static constexpr uint8_t  ICA_CHANNELS     = 3;     ///< Fixed 3-source model
static constexpr uint16_t ICA_BLOCK_SIZE   = 512;   ///< Samples per ICA frame
static constexpr uint8_t  ICA_MAX_ITER     = 100;   ///< Max Newton iterations
static constexpr float    ICA_TOL          = 1e-5f; ///< Convergence threshold
static constexpr float    ICA_ALPHA        = 1.0f;  ///< tanh nonlinearity scale

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * @brief  Main entry point. Processes one 3×512 ADC frame through FastICA.
 *
 * @param  raw_in    Input: raw[channel][sample] — uint16_t ADC readings (0–4095)
 * @param  sources   Output: sources[component][sample] — float32 demixed signals
 * @param  num_ch    Number of channels (must equal ICA_CHANNELS = 3)
 * @param  num_samp  Samples per channel (must equal ICA_BLOCK_SIZE = 512)
 *
 * @note   Thread safety: Not ISR-safe. Call only from main loop after
 *         noInterrupts()/interrupts() guard around the input copy.
 */
void dsp_fastica_process(
    const uint16_t raw_in[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float          sources[ICA_CHANNELS][ICA_BLOCK_SIZE],
    uint8_t        num_ch,
    uint16_t       num_samp
);

// ── Internal Helper Declarations (exposed for unit testing) ─────────────────

/**
 * @brief Converts uint16_t ADC values to centered, scaled float32.
 *        Subtracts per-channel mean. Scales to approximate microvolt range
 *        assuming 3.3V reference, 12-bit ADC: 1 LSB ≈ 0.8057 µV.
 */
void dsp_adc_to_float(
    const uint16_t in[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float          out[ICA_CHANNELS][ICA_BLOCK_SIZE]
);

/**
 * @brief Computes the 3×3 covariance matrix of the input signal matrix.
 * @param X     Input float matrix [channels × samples]
 * @param cov   Output 3×3 covariance matrix (row-major)
 */
void dsp_compute_covariance(
    const float X[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       cov[ICA_CHANNELS][ICA_CHANNELS]
);

/**
 * @brief Computes eigenvalues and eigenvectors of a symmetric 3×3 matrix
 *        using Jacobi iteration (suitable for small fixed-size matrices).
 * @param A         Input symmetric 3×3 matrix (modified in-place)
 * @param eigvals   Output eigenvalue vector [3]
 * @param eigvecs   Output eigenvector matrix [3×3], column = eigenvector
 */
void dsp_eigen3x3(
    float A[ICA_CHANNELS][ICA_CHANNELS],
    float eigvals[ICA_CHANNELS],
    float eigvecs[ICA_CHANNELS][ICA_CHANNELS]
);

/**
 * @brief Builds the 3×3 whitening matrix W_white = Λ^(-½) · Eᵀ and
 *        applies it to produce the whitened observation matrix Z.
 * @param X         Input: centered observation matrix [channels × samples]
 * @param Z         Output: whitened matrix [channels × samples]
 * @param W_white   Output: 3×3 whitening matrix (for later inverse transform)
 */
void dsp_whiten(
    const float X[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       Z[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       W_white[ICA_CHANNELS][ICA_CHANNELS]
);

/**
 * @brief One FastICA weight vector update step (deflation scheme).
 *        Updates w using the tanh nonlinearity:
 *            w ← E[z·tanh(wᵀz)] − E[α(1−tanh²(wᵀz))]·w
 *            w ← w / ‖w‖
 * @param Z      Whitened matrix [channels × samples]
 * @param w      Weight vector [channels] — updated in-place
 * @return       true if converged (Δw < ICA_TOL), false otherwise
 */
bool dsp_ica_weight_update(
    const float Z[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       w[ICA_CHANNELS]
);

/**
 * @brief Gram-Schmidt deflation: orthogonalizes w against all previously
 *        extracted weight vectors to ensure independent components.
 * @param W_prev   Previously extracted weight vectors [comp_idx × channels]
 * @param w        Current weight vector — deflated in-place
 * @param num_prev Number of already-extracted components
 */
void dsp_deflate(
    const float W_prev[ICA_CHANNELS][ICA_CHANNELS],
    float       w[ICA_CHANNELS],
    uint8_t     num_prev
);
