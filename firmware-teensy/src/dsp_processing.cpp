/**
 * @file    dsp_processing.cpp
 * @brief   SSI DSP Layer — FastICA Implementation (float32, Cortex-M7 FPU)
 *
 * Full implementation of the FastICA pipeline declared in dsp_processing.h.
 * All matrix operations are performed in float32 using standard C math
 * functions accelerated by the Teensy 4.1's VFPv5-D16 hardware FPU.
 *
 * Pipeline stages per frame:
 *   1. ADC → float32 conversion + mean centering
 *   2. Covariance matrix computation  (3×3)
 *   3. Eigendecomposition via Jacobi  (3×3 symmetric)
 *   4. Whitening transform            (Z = Λ^(-½)Eᵀ · X)
 *   5. Iterative weight extraction    (deflation scheme, tanh nonlinearity)
 *   6. Source reconstruction          (S = W · Z)
 */

#include "dsp_processing.h"
#include <math.h>
#include <string.h>

// ── Internal Utility Macros ──────────────────────────────────────────────────
#define IDX2D(r, c, ncols)  ((r) * (ncols) + (c))
#define SQ(x)               ((x) * (x))
#define CLAMP(v, lo, hi)    ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

// ADC LSB → microvolt conversion factor
// Vref = 3.3V, 12-bit = 4096 levels → 3.3/4096 * 1e6 µV ≈ 805.7 µV/LSB
static constexpr float ADC_UV_PER_LSB = 805.7f;
static constexpr float ADC_MIDPOINT   = 2048.0f; // 12-bit center

// ── 1. ADC → Float Conversion + Mean Centering ──────────────────────────────
/**
 * Converts raw 12-bit ADC readings to centered float32 microvolt values.
 *
 * Step 1: offset by ADC midpoint and scale to µV
 *   out[ch][i] = (in[ch][i] - midpoint) * UV_PER_LSB
 *
 * Step 2: subtract per-channel sample mean
 *   µ_ch = (1/N) Σ out[ch][i]
 *   out[ch][i] -= µ_ch
 */
void dsp_adc_to_float(
    const uint16_t in[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float          out[ICA_CHANNELS][ICA_BLOCK_SIZE]
) {
    for (uint8_t ch = 0; ch < ICA_CHANNELS; ch++) {
        // Convert to µV with midpoint offset
        float sum = 0.0f;
        for (uint16_t i = 0; i < ICA_BLOCK_SIZE; i++) {
            float v = (static_cast<float>(in[ch][i]) - ADC_MIDPOINT) * ADC_UV_PER_LSB;
            out[ch][i] = v;
            sum += v;
        }
        // Subtract mean (centering step — required for ICA covariance validity)
        float mean = sum / static_cast<float>(ICA_BLOCK_SIZE);
        for (uint16_t i = 0; i < ICA_BLOCK_SIZE; i++) {
            out[ch][i] -= mean;
        }
    }
}

// ── 2. Covariance Matrix ─────────────────────────────────────────────────────
/**
 * Computes the 3×3 sample covariance matrix:
 *
 *   Σ[p][q] = (1/(N-1)) Σ_i X[p][i] · X[q][i]
 *
 * Input X is already mean-centered (from dsp_adc_to_float).
 * Output cov is symmetric positive semi-definite.
 */
void dsp_compute_covariance(
    const float X[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       cov[ICA_CHANNELS][ICA_CHANNELS]
) {
    const float inv_N1 = 1.0f / static_cast<float>(ICA_BLOCK_SIZE - 1);

    for (uint8_t p = 0; p < ICA_CHANNELS; p++) {
        for (uint8_t q = p; q < ICA_CHANNELS; q++) {
            float acc = 0.0f;
            for (uint16_t i = 0; i < ICA_BLOCK_SIZE; i++) {
                acc += X[p][i] * X[q][i];
            }
            cov[p][q] = acc * inv_N1;
            cov[q][p] = cov[p][q]; // Symmetric
        }
    }
}

// ── 3. Jacobi Eigendecomposition (3×3 Symmetric) ────────────────────────────
/**
 * Computes eigenvalues and eigenvectors of a real symmetric 3×3 matrix A
 * using the classical Jacobi iteration method.
 *
 * Algorithm:
 *   Repeat until off-diagonal elements < tolerance:
 *     - Find largest off-diagonal element A[p][q]
 *     - Compute Givens rotation angle θ: cot(2θ) = (A[q][q]-A[p][p])/(2·A[p][q])
 *     - Apply Givens rotation: A ← Rᵀ·A·R, V ← V·R
 *   Eigenvalues = diagonal of A, Eigenvectors = columns of V
 *
 * Convergence: ~O(n³ log(1/ε)) for n×n matrix, very fast for n=3.
 *
 * @param A       Symmetric 3×3 matrix, modified in-place → diagonal eigenvalues
 * @param eigvals Extracted eigenvalues
 * @param eigvecs Column i = eigenvector for eigvals[i]
 */
void dsp_eigen3x3(
    float A[ICA_CHANNELS][ICA_CHANNELS],
    float eigvals[ICA_CHANNELS],
    float eigvecs[ICA_CHANNELS][ICA_CHANNELS]
) {
    // Initialize eigvecs to identity
    for (uint8_t i = 0; i < ICA_CHANNELS; i++) {
        for (uint8_t j = 0; j < ICA_CHANNELS; j++) {
            eigvecs[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    static constexpr int   MAX_JACOBI_ITER = 50;
    static constexpr float JACOBI_TOL      = 1e-8f;

    for (int iter = 0; iter < MAX_JACOBI_ITER; iter++) {
        // Find largest off-diagonal element
        float max_off = 0.0f;
        uint8_t p = 0, q = 1;
        for (uint8_t r = 0; r < ICA_CHANNELS; r++) {
            for (uint8_t s = r + 1; s < ICA_CHANNELS; s++) {
                float abv = fabsf(A[r][s]);
                if (abv > max_off) { max_off = abv; p = r; q = s; }
            }
        }

        if (max_off < JACOBI_TOL) break; // Converged

        // Compute Givens rotation angle
        float theta;
        float denom = A[q][q] - A[p][p];
        if (fabsf(denom) < 1e-12f) {
            theta = (float)M_PI / 4.0f; // 45° when denominator ≈ 0
        } else {
            theta = 0.5f * atanf(2.0f * A[p][q] / denom);
        }
        float c = cosf(theta);
        float s = sinf(theta);

        // Apply Givens rotation to A (in-place symmetric update)
        float App = c*c*A[p][p] - 2.0f*s*c*A[p][q] + s*s*A[q][q];
        float Aqq = s*s*A[p][p] + 2.0f*s*c*A[p][q] + c*c*A[q][q];
        float Apq = 0.0f; // Zeroed by rotation (by construction)

        // Update off-diagonal rows/cols r ≠ p, q
        float Arp[ICA_CHANNELS], Arq[ICA_CHANNELS];
        for (uint8_t r = 0; r < ICA_CHANNELS; r++) {
            if (r == p || r == q) continue;
            Arp[r] =  c * A[r][p] - s * A[r][q];
            Arq[r] =  s * A[r][p] + c * A[r][q];
        }
        for (uint8_t r = 0; r < ICA_CHANNELS; r++) {
            if (r == p || r == q) continue;
            A[r][p] = A[p][r] = Arp[r];
            A[r][q] = A[q][r] = Arq[r];
        }
        A[p][p] = App; A[q][q] = Aqq; A[p][q] = A[q][p] = Apq;

        // Accumulate rotation into eigvec matrix
        for (uint8_t r = 0; r < ICA_CHANNELS; r++) {
            float vp =  c * eigvecs[r][p] - s * eigvecs[r][q];
            float vq =  s * eigvecs[r][p] + c * eigvecs[r][q];
            eigvecs[r][p] = vp;
            eigvecs[r][q] = vq;
        }
    }

    // Extract diagonal eigenvalues
    for (uint8_t i = 0; i < ICA_CHANNELS; i++) {
        eigvals[i] = A[i][i];
    }
}

// ── 4. Whitening Transform ───────────────────────────────────────────────────
/**
 * Constructs whitening matrix W_white = Λ^(-½) · Eᵀ from eigendecomposition
 * of the covariance matrix Σ, then applies it:
 *
 *     Z = W_white · X    →   E[ZZᵀ] = I  (unit covariance, uncorrelated)
 *
 * Λ^(-½)[i][i] = 1 / sqrt(λ_i + ε)   where ε = 1e-6 prevents divide-by-zero
 *
 * @param X       Mean-centered input matrix [channels × samples]
 * @param Z       Output whitened matrix     [channels × samples]
 * @param W_white Output 3×3 whitening matrix (stored for reference/inverse)
 */
void dsp_whiten(
    const float X[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       Z[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       W_white[ICA_CHANNELS][ICA_CHANNELS]
) {
    static constexpr float EPS = 1e-6f;

    // Step 1: Compute covariance
    float cov[ICA_CHANNELS][ICA_CHANNELS];
    dsp_compute_covariance(X, cov);

    // Step 2: Eigendecompose covariance
    float eigvals[ICA_CHANNELS];
    float eigvecs[ICA_CHANNELS][ICA_CHANNELS];
    // Jacobi modifies A in-place; pass a copy
    float cov_copy[ICA_CHANNELS][ICA_CHANNELS];
    memcpy(cov_copy, cov, sizeof(cov_copy));
    dsp_eigen3x3(cov_copy, eigvals, eigvecs);

    // Step 3: Build W_white = Λ^(-½) · Eᵀ
    // W_white[i][j] = (1/sqrt(λ_i)) * eigvecs[j][i]  (transpose: col→row)
    for (uint8_t i = 0; i < ICA_CHANNELS; i++) {
        float inv_sqrt_lam = 1.0f / sqrtf(fabsf(eigvals[i]) + EPS);
        for (uint8_t j = 0; j < ICA_CHANNELS; j++) {
            W_white[i][j] = inv_sqrt_lam * eigvecs[j][i];
        }
    }

    // Step 4: Apply Z = W_white · X
    for (uint8_t i = 0; i < ICA_CHANNELS; i++) {
        for (uint16_t n = 0; n < ICA_BLOCK_SIZE; n++) {
            float acc = 0.0f;
            for (uint8_t k = 0; k < ICA_CHANNELS; k++) {
                acc += W_white[i][k] * X[k][n];
            }
            Z[i][n] = acc;
        }
    }
}

// ── 5a. FastICA Weight Update ────────────────────────────────────────────────
/**
 * Performs one Newton-Raphson weight update for extracting one independent
 * component from the whitened observation matrix Z.
 *
 * Given weight vector w, the update rule is:
 *
 *   E1 = (1/N) Σ_n  z_n · tanh(α · wᵀz_n)
 *   E2 = (1/N) Σ_n  α · (1 − tanh²(α · wᵀz_n))
 *
 *   w_new = E1 − E2 · w
 *   w_new = w_new / ‖w_new‖   (L2 normalization)
 *
 * Convergence is measured as: 1 − |w_new · w| < ICA_TOL
 *
 * @param Z   Whitened 3×N matrix
 * @param w   Weight vector [ICA_CHANNELS], updated in-place
 * @return    true if ‖w_new − w‖ < ICA_TOL (converged)
 */
bool dsp_ica_weight_update(
    const float Z[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float       w[ICA_CHANNELS]
) {
    float E1[ICA_CHANNELS] = {0.0f, 0.0f, 0.0f};
    float E2 = 0.0f;

    for (uint16_t n = 0; n < ICA_BLOCK_SIZE; n++) {
        // Compute projection: u = wᵀ·z_n
        float u = 0.0f;
        for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
            u += w[c] * Z[c][n];
        }

        // g(u) = tanh(α·u),  g′(u) = α·(1 − tanh²(α·u))
        float tanh_u = tanhf(ICA_ALPHA * u);
        float g_u    = tanh_u;
        float gp_u   = ICA_ALPHA * (1.0f - tanh_u * tanh_u);

        // Accumulate E1 and E2
        for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
            E1[c] += Z[c][n] * g_u;
        }
        E2 += gp_u;
    }

    const float inv_N = 1.0f / static_cast<float>(ICA_BLOCK_SIZE);
    E2 *= inv_N;

    // w_new = E1/N − E2·w
    float w_new[ICA_CHANNELS];
    float norm_sq = 0.0f;
    for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
        w_new[c] = E1[c] * inv_N - E2 * w[c];
        norm_sq += w_new[c] * w_new[c];
    }

    // Normalize w_new
    float inv_norm = 1.0f / (sqrtf(norm_sq) + 1e-12f);
    for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
        w_new[c] *= inv_norm;
    }

    // Convergence check: 1 − |w_new · w_old| < TOL
    float dot = 0.0f;
    for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
        dot += w_new[c] * w[c];
    }
    bool converged = (1.0f - fabsf(dot)) < ICA_TOL;

    // Update w
    for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
        w[c] = w_new[c];
    }

    return converged;
}

// ── 5b. Gram-Schmidt Deflation ───────────────────────────────────────────────
/**
 * Projects w onto the subspace orthogonal to all previously extracted
 * weight vectors (deflation step in the FastICA deflation scheme).
 *
 *   w ← w − Σ_{j<k} (wᵀ·w_j) · w_j
 *   w ← w / ‖w‖
 *
 * Ensures extracted ICA components are linearly independent.
 */
void dsp_deflate(
    const float W_prev[ICA_CHANNELS][ICA_CHANNELS],
    float       w[ICA_CHANNELS],
    uint8_t     num_prev
) {
    for (uint8_t j = 0; j < num_prev; j++) {
        // Compute dot product w · W_prev[j]
        float dot = 0.0f;
        for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
            dot += w[c] * W_prev[j][c];
        }
        // Subtract projection
        for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
            w[c] -= dot * W_prev[j][c];
        }
    }

    // Re-normalize
    float norm_sq = 0.0f;
    for (uint8_t c = 0; c < ICA_CHANNELS; c++) norm_sq += w[c] * w[c];
    float inv_norm = 1.0f / (sqrtf(norm_sq) + 1e-12f);
    for (uint8_t c = 0; c < ICA_CHANNELS; c++) w[c] *= inv_norm;
}

// ── 6. Main Entry Point ──────────────────────────────────────────────────────
/**
 * Full FastICA pipeline for one 3×512 calibration frame.
 *
 * Execution order:
 *   (a) ADC → centered float32
 *   (b) Whitening
 *   (c) FastICA deflation loop for ICA_CHANNELS components
 *   (d) Source reconstruction: S = W_demix · Z
 *
 * Timing estimate (Cortex-M7 @ 600 MHz, -O3):
 *   Steps a-b: ~0.1 ms (SIMD-vectorized FPU ops)
 *   Step c:    ~1–3 ms (100 Newton iterations × 3 components × 512 samples)
 *   Total:     <5 ms per frame → real-time safe at 3.9 Hz frame rate
 */
void dsp_fastica_process(
    const uint16_t raw_in[ICA_CHANNELS][ICA_BLOCK_SIZE],
    float          sources[ICA_CHANNELS][ICA_BLOCK_SIZE],
    uint8_t        num_ch,
    uint16_t       num_samp
) {
    (void)num_ch;   // Statically fixed to ICA_CHANNELS
    (void)num_samp; // Statically fixed to ICA_BLOCK_SIZE

    // (a) Convert ADC → centered float32
    float X[ICA_CHANNELS][ICA_BLOCK_SIZE];
    dsp_adc_to_float(raw_in, X);

    // (b) Whiten
    float Z[ICA_CHANNELS][ICA_BLOCK_SIZE];
    float W_white[ICA_CHANNELS][ICA_CHANNELS];
    dsp_whiten(X, Z, W_white);

    // (c) Extract ICA_CHANNELS independent components via deflation
    float W_demix[ICA_CHANNELS][ICA_CHANNELS]; // Row k = demixing vector for source k
    memset(W_demix, 0, sizeof(W_demix));

    for (uint8_t comp = 0; comp < ICA_CHANNELS; comp++) {
        // Initialize weight vector to unit basis vector e_comp
        float w[ICA_CHANNELS] = {0.0f, 0.0f, 0.0f};
        w[comp] = 1.0f;

        // Iterative Newton-Raphson update
        for (uint8_t iter = 0; iter < ICA_MAX_ITER; iter++) {
            // Gram-Schmidt orthogonalization vs. previous components
            if (comp > 0) {
                dsp_deflate(W_demix, w, comp);
            }

            bool converged = dsp_ica_weight_update(Z, w);
            if (converged) break;
        }

        // Final deflation pass
        if (comp > 0) {
            dsp_deflate(W_demix, w, comp);
        }

        // Store extracted weight vector
        for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
            W_demix[comp][c] = w[c];
        }
    }

    // (d) Source reconstruction: S[comp][n] = Σ_c W_demix[comp][c] · Z[c][n]
    for (uint8_t comp = 0; comp < ICA_CHANNELS; comp++) {
        for (uint16_t n = 0; n < ICA_BLOCK_SIZE; n++) {
            float acc = 0.0f;
            for (uint8_t c = 0; c < ICA_CHANNELS; c++) {
                acc += W_demix[comp][c] * Z[c][n];
            }
            sources[comp][n] = acc;
        }
    }
}
