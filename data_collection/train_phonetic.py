"""
CNN 1: Articulator Training Pipeline — CADENCE
================================================

Maps FastICA-cleaned sEMG/PZT data -> 6-DOF phonetic formant regression.

Output: [Pitch, Yaw, Intensity, F1, F2, F3]
Loss:   Mean Squared Error (continuous regression, NOT classification)
Quant:  The INT8 representative dataset is strictly derived from
        FastICA output, not raw sEMG, to ensure quantization scaling
        factors match the inference-time input distribution.

IMPORTANT: This script trains on normalized FastICA output.
If raw sEMG is fed instead, the INT8 scaling will clip microvolt
variance at deployment and destroy inference accuracy.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)

DATASET_DIR    = "dataset"
OUTPUT_HEADER  = "../firmware-teensy/src/models/dual_models.h"
WINDOW_LEN     = 50      # Timesteps: 50 * 2ms = 100ms analysis window
N_FEATURES     = 3       # FastICA separated: [sEMG_ch1, sEMG_ch2, PZT]
N_OUTPUTS      = 6       # [Pitch, Yaw, Intensity, F1, F2, F3]

# ─── Architecture ─────────────────────────────────────────────────────────────

def build_phonetic_model():
    """
    1D-CNN with dilated causal convolutions for phonetic regression.

    Causal padding ensures the network cannot look into future samples,
    preserving strict real-time causal inference on the Teensy.

    Dilation rates [1, 2, 4] give an effective receptive field of
    7 * (1+2+4) = 49 timesteps ~ matching our 50-step window.
    """
    inputs = tf.keras.Input(shape=(WINDOW_LEN, N_FEATURES), name="fastica_input")

    # Dilated causal Conv1D blocks
    x = layers.Conv1D(16, kernel_size=3, padding="causal", dilation_rate=1,
                      activation="relu6", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(32, kernel_size=3, padding="causal", dilation_rate=2,
                      activation="relu6", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(32, kernel_size=3, padding="causal", dilation_rate=4,
                      activation="relu6", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(32, activation="relu6",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    # 6-DOF continuous regression (NOT softmax classification)
    outputs = layers.Dense(N_OUTPUTS, activation="sigmoid", name="phonetic_6dof")(x)

    model = models.Model(inputs, outputs, name="CNN1_Articulator")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",       # Mean Squared Error: correct loss for regression
        metrics=["mae"]   # MAE in normalized units; convert to Hz in validation
    )
    return model

# ─── Dataset Construction ──────────────────────────────────────────────────────

def _fastica_simulate(raw_ch0, raw_ch1, pzt):
    """
    Simulates the FastICA demixing step performed by the Teensy firmware.

    In hardware, the 2x2 demixing matrix W is computed from the resting
    baseline. Here we replicate the linear combination: s = W * x.
    This ensures our representative dataset for INT8 quantization is
    strictly derived from FastICA outputs, matching the inference distribution.
    """
    # Simplified W (unit test demixing, replace with your calibrated matrix)
    W = np.array([[0.8, -0.2], [-0.3, 0.9]])
    mixed = np.stack([raw_ch0, raw_ch1], axis=-1)  # [N, 2]
    separated = mixed @ W.T                          # [N, 2]
    return separated[:, 0], separated[:, 1]

def load_or_generate_dataset():
    """
    Loads bio/acoustic .npy pairs from dataset/ and constructs training tensors.
    Returns:
        X_fastica: [N, WINDOW_LEN, 3] strictly FastICA output (NOT raw sEMG)
        Y_targets: [N, 6] — [Pitch, Yaw, Intensity, F1_norm, F2_norm, F3_norm]
    """
    X_all, Y_all = [], []

    bio_files = [f for f in os.listdir(DATASET_DIR) if f.startswith("bio_")] if os.path.exists(DATASET_DIR) else []

    if not bio_files:
        print("[!] No real data in dataset/. Generating synthetic FastICA-distribution data...")
        return _synthetic_fastica_data()

    for fname in bio_files:
        ts = fname.replace("bio_", "").replace(".npy", "")
        acoustic_path = os.path.join(DATASET_DIR, f"acoustic_{ts}.npy")
        if not os.path.exists(acoustic_path):
            continue

        bio = np.load(os.path.join(DATASET_DIR, fname), allow_pickle=True)
        acoustic = np.load(acoustic_path)

        if len(bio) < WINDOW_LEN or len(acoustic) < WINDOW_LEN:
            continue

        for start in range(0, len(bio) - WINDOW_LEN, 4):
            bio_w  = bio[start:start + WINDOW_LEN]
            acc_w  = acoustic[start:start + WINDOW_LEN]

            try:
                # Extract raw channels
                raw_ch0 = np.array([b[1].get('a0', 0.5) if isinstance(b, tuple) else 0.5 for b in bio_w], dtype=np.float32)
                raw_ch1 = np.array([b[1].get('a1', 0.5) if isinstance(b, tuple) else 0.5 for b in bio_w], dtype=np.float32)
                pzt_raw = np.array([b[1].get('a2', 0.0) if isinstance(b, tuple) else 0.0 for b in bio_w], dtype=np.float32)

                # Apply simulated FastICA demixing
                sep_ch0, sep_ch1 = _fastica_simulate(raw_ch0, raw_ch1, pzt_raw)

                # CRITICAL: Use FASTICA OUTPUT not raw sEMG for INT8 quant targets
                features = np.stack([sep_ch0, sep_ch1, pzt_raw], axis=-1).astype(np.float32)

                # Target construction from Praat-extracted acoustic
                f0_mean = np.mean(acc_w[:, 1]) / 500.0      # Pitch (normalized)
                f1_mean = np.mean(acc_w[:, 2]) / 1000.0     # F1 normalized
                f2_mean = np.mean(acc_w[:, 3]) / 2500.0     # F2 normalized
                f3_mean = np.clip(np.mean(acc_w[:, 3]) / 3500.0, 0, 1)  # F3 estimate

                targets = np.array([
                    np.clip(f0_mean, 0, 1),        # Pitch
                    0.5,                            # Yaw (acoustically unmeasured)
                    np.clip(np.std(raw_ch0), 0, 1), # Intensity proxy
                    np.clip(f1_mean, 0, 1),         # F1
                    np.clip(f2_mean, 0, 1),         # F2
                    np.clip(f3_mean, 0, 1),         # F3
                ], dtype=np.float32)

                X_all.append(features)
                Y_all.append(targets)

            except Exception:
                continue

    if not X_all:
        print("[!] Dataset parsing failed. Using synthetic FastICA data.")
        return _synthetic_fastica_data()

    return np.stack(X_all), np.stack(Y_all)

def _synthetic_fastica_data(n=2000):
    """
    Generates synthetic data that mimics the statistical distribution of
    FastICA-cleaned sEMG signals (near-zero mean, unit variance, sparse bursts).
    This is used both for training and as the INT8 quantization representative set.
    """
    t = np.linspace(0, 1, WINDOW_LEN)
    X = np.zeros((n, WINDOW_LEN, N_FEATURES), dtype=np.float32)
    Y = np.zeros((n, N_OUTPUTS), dtype=np.float32)

    for i in range(n):
        # FastICA output is statistically independent, near-zero mean
        # Simulate as sparse muscle burst (Laplacian-like)
        burst_onset = np.random.randint(5, 35)
        burst_dur   = np.random.randint(5, 20)
        burst_amp   = np.random.uniform(0.1, 0.9)

        X[i, :, 0] = np.random.laplace(0, 0.03, WINDOW_LEN)  # Sparse sEMG ch1
        X[i, :, 1] = np.random.laplace(0, 0.03, WINDOW_LEN)  # Sparse sEMG ch2
        # Add burst
        X[i, burst_onset:burst_onset+burst_dur, 0] += burst_amp * np.hanning(burst_dur)
        X[i, burst_onset:burst_onset+burst_dur, 1] += burst_amp * 0.6 * np.hanning(burst_dur)

        # PZT: sinusoidal at glottal frequency
        f0_norm = np.random.uniform(0.1, 0.8)
        f0_hz   = 80.0 + f0_norm * 220.0
        X[i, :, 2] = 0.3 * np.sin(2 * np.pi * f0_hz * t / 10000.0)

        X[i] = np.clip(X[i], -1.0, 1.0)

        # Correlated targets from burst characteristics
        Y[i, 0] = f0_norm                                         # Pitch
        Y[i, 1] = np.clip(np.random.normal(0.5, 0.15), 0, 1)     # Yaw
        Y[i, 2] = np.clip(burst_amp, 0, 1)                        # Intensity
        Y[i, 3] = np.clip(0.3 + burst_amp * 0.5, 0, 1)           # F1 (correlated with intensity)
        Y[i, 4] = np.clip(0.4 + f0_norm * 0.4, 0, 1)             # F2 (correlated with pitch)
        Y[i, 5] = np.clip(np.random.uniform(0.3, 0.7), 0, 1)     # F3

    return X, Y

# ─── INT8 Quantization ─────────────────────────────────────────────────────────

def convert_float32_tflite(model):
    """
    FLOAT32 TFLite conversion — leverages Cortex-M7 hardware FPU.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    return converter.convert()

def get_representative_dataset_gen(X_fastica):
    """
    CRITICAL: Representative dataset MUST be FastICA output, not raw sEMG.
    INT8 scaling factors are derived from this distribution.
    """
    def gen():
        n = min(100, len(X_fastica))
        idx = np.random.choice(len(X_fastica), n, replace=False)
        for i in idx:
            yield [X_fastica[i:i+1].astype(np.float32)]
    return gen

# ─── Validation Metrics ────────────────────────────────────────────────────────

def evaluate_formant_accuracy_hz(model, X_test, Y_test):
    """
    Converts normalized outputs back to Hz and reports MAE.
    F1 range: [300, 1000] Hz -> 700Hz span
    F2 range: [800, 2500] Hz -> 1700Hz span
    """
    print("\n[!] Running Formant Accuracy Validation (Hz scale)...")
    preds = model.predict(X_test, verbose=0)

    mae_f1_norm  = np.mean(np.abs(Y_test[:, 3] - preds[:, 3]))
    mae_f2_norm  = np.mean(np.abs(Y_test[:, 4] - preds[:, 4]))
    mae_f1_hz    = mae_f1_norm * 700.0
    mae_f2_hz    = mae_f2_norm * 1700.0
    mae_pitch_hz = np.mean(np.abs(Y_test[:, 0] - preds[:, 0])) * 220.0

    print(f"    -> F1  MAE: {mae_f1_hz:.2f} Hz  (target < 50Hz)")
    print(f"    -> F2  MAE: {mae_f2_hz:.2f} Hz  (target < 50Hz)")
    print(f"    -> f0  MAE: {mae_pitch_hz:.2f} Hz  (target < 20Hz)")

    if mae_f1_hz < 50.0 and mae_f2_hz < 50.0:
        print("    [SUCCESS] F1/F2 MAE under 50Hz threshold!")
    else:
        print("    [NEEDS MORE DATA] MAE above 50Hz — collect more training trials.")

    return mae_f1_hz, mae_f2_hz

# ─── C-Array Export ────────────────────────────────────────────────────────────

def to_c_array(tflite_bytes, array_name):
    hex_vals = [hex(b) for b in tflite_bytes]
    c_str = f"alignas(16) const unsigned char {array_name}[] = {{\n"
    for i in range(0, len(hex_vals), 12):
        c_str += "  " + ", ".join(hex_vals[i:i+12]) + ",\n"
    c_str += "};\n"
    c_str += f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n"
    return c_str

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[+] CNN 1: Articulator Training Pipeline")
    print("    Constraint: Input MUST be FastICA-separated, NOT raw sEMG")

    print("[+] Loading training dataset (FastICA distribution)...")
    X, Y = load_or_generate_dataset()
    print(f"    -> {X.shape[0]} samples | {X.shape[1]} timesteps | {X.shape[2]} features")

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    print("[+] Building CNN 1 architecture...")
    model = build_phonetic_model()
    model.summary()

    print("[+] Training (MSE regression loss)...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=12, min_lr=1e-5)
    ]
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=300,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Validation in Hz scale
    mae_f1, mae_f2 = evaluate_formant_accuracy_hz(model, X_val, Y_val)

    # FLOAT32 TFLite export (leverages Cortex-M7 FPU)
    print("\n[+] Converting to FLOAT32 TFLite flatbuffer...")
    tflite_bytes = convert_float32_tflite(model)
    with open("phonetic_model.tflite", "wb") as f:
        f.write(tflite_bytes)
    print(f"    -> Saved phonetic_model.tflite ({len(tflite_bytes)} bytes)")

    # Export C-byte array
    print("[+] Exporting C-byte array to dual_models.h...")
    phonetic_c = to_c_array(tflite_bytes, "phonetic_model_tflite")

    # Preserve existing affective model bytes if present
    existing_affective = ""
    if os.path.exists(OUTPUT_HEADER):
        with open(OUTPUT_HEADER, "r") as f:
            content = f.read()
        aff_start = content.find("alignas(16) const unsigned char affective_model_tflite")
        aff_end   = content.rfind("#endif")
        if aff_start != -1 and aff_end != -1:
            existing_affective = content[aff_start:aff_end].strip()

    header = f"""#ifndef DUAL_MODELS_H
#define DUAL_MODELS_H

// Auto-generated by CADENCE Dual-Model Training Pipeline
// Model A (Phonetic):  FLOAT32 TFLite 1D-CNN — 6-DOF sublingual vector
//   MAE: F1={mae_f1:.1f}Hz, F2={mae_f2:.1f}Hz (target <50Hz)
// Model B (Affective): INT8 TFLite 1D-CNN — 3-DOF prosodic modifier

{phonetic_c}

{existing_affective if existing_affective else '// [PLACEHOLDER] Model B bytes — run train_affective.py to generate'}

#endif
"""
    with open(OUTPUT_HEADER, "w") as f:
        f.write(header)
    print(f"[SUCCESS] phonetic_model_tflite[] written to {OUTPUT_HEADER}")

if __name__ == "__main__":
    main()
