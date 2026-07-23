"""
CNN 2: Affective Modulator — Training & INT8 Quantization Pipeline
Subvocal Synthesis Interface (CADENCE)

Input:  4-DOF physiological feature vector [f0, RMS, MNF, MDF] over N time steps
Output: 3-DOF prosodic modifier [Pitch_Shift, Volume_Gain, Tempo_Modifier]

Architecture: Depthwise Separable 1D-CNN (minimized MAC operations for Cortex-M7)
Quantization: INT8 Post-Training Quantization (PTQ) via TFLite converter
Export:       C-byte array to dual_models.h affective_model_tflite[] field

Execution budget on Teensy 4.1: < 150us (within global 450us clamp)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)

DATASET_DIR = "dataset"
OUTPUT_HEADER = "../firmware-teensy/src/models/dual_models.h"

# ─── Model Architecture ────────────────────────────────────────────────────────

def depthwise_sep_block(x, filters, kernel_size):
    """
    Depthwise Separable Convolution block.
    Dramatically reduces MAC operations vs standard Conv1D.
    Depthwise: kernels act on each feature independently (spatial pattern)
    Pointwise: 1x1 conv mixes channels (feature interaction)
    """
    x = layers.DepthwiseConv1D(
        kernel_size=kernel_size,
        padding="causal",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Conv1D(filters, kernel_size=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)  # ReLU6 — improves INT8 quantization
    return x

def build_affective_model(n_timesteps=32, n_features=4):
    """
    Hyper-lightweight 1D-CNN for physiological prosodic modulation.

    Designed specifically to minimize multiply-accumulate (MAC) operations
    on the ARM Cortex-M7. ReLU6 is used over standard ReLU to improve
    the numerical stability of INT8 quantization.

    Input:  [1, 32, 4] — 32 timesteps of [f0, RMS, MNF, MDF] features
    Output: [1, 3]    — [Pitch_Shift, Volume_Gain, Tempo_Modifier] in [0.0, 1.0]
    """
    inputs = tf.keras.Input(shape=(n_timesteps, n_features), name="physiological_input")

    # Depthwise separable block 1 — local short-range patterns
    x = depthwise_sep_block(inputs, filters=8, kernel_size=3)

    # Depthwise separable block 2 — wider temporal context
    x = depthwise_sep_block(x, filters=16, kernel_size=5)

    # Global Average Pooling: summarizes whole sequence into a single vector
    # This is preferable to Flatten for fixed small MAC budget
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Single dense layer with L2 regularization
    x = layers.Dense(16, activation="relu6", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    # Output: 3-DOF prosodic modifier
    outputs = layers.Dense(3, activation="sigmoid", name="prosodic_output")(x)

    model = models.Model(inputs, outputs, name="CNN2_Affective_Modulator")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ─── Dataset Loading ───────────────────────────────────────────────────────────

def load_affective_dataset(dataset_dir, n_timesteps=32):
    """
    Loads aligned dataset from .npy files produced by collect.py.
    Constructs sliding-window sequences for temporal CNN input.

    Returns:
        X: [N, n_timesteps, 4] float32 input tensors
        Y: [N, 3] float32 target tensors (Pitch_Shift, Volume_Gain, Tempo)
    """
    X_all, Y_all = [], []

    if not os.path.exists(dataset_dir):
        print(f"[!] Dataset dir '{dataset_dir}' not found. Using synthetic data.")
        return _generate_synthetic_data(n_timesteps)

    for fname in os.listdir(dataset_dir):
        if not fname.startswith("bio_"): continue
        bio_path = os.path.join(dataset_dir, fname)
        timestamp = fname.replace("bio_", "").replace(".npy", "")
        acoustic_path = os.path.join(dataset_dir, f"acoustic_{timestamp}.npy")

        if not os.path.exists(acoustic_path):
            continue

        bio = np.load(bio_path, allow_pickle=True)
        acoustic = np.load(acoustic_path)

        if bio.shape[0] < n_timesteps or acoustic.shape[0] < n_timesteps:
            continue

        for start in range(0, len(bio) - n_timesteps, 4):
            bio_window = bio[start:start + n_timesteps]
            acc_window = acoustic[start:start + n_timesteps]

            # Feature sequence: [f0, RMS, MNF, MDF] per timestep
            # For bio data: col 0 = ch1_emg, col 1 = ch2_emg, col 2 = pzt, col 3 = eda
            # Acoustic: col 0 = time, col 1 = f0, col 2 = F1, col 3 = F2
            try:
                features = np.stack([
                    acc_window[:, 1] / 500.0,      # f0 normalized
                    np.abs(bio_window[:, 2]),        # PZT RMS (normalized by firmware)
                    np.abs(bio_window[:, 0]),        # sEMG ch1 (used as MNF proxy)
                    np.abs(bio_window[:, 1]),        # sEMG ch2 (used as MDF proxy)
                ], axis=-1).astype(np.float32)

                # Targets: map acoustic features to prosodic modifiers
                f0_mean = np.mean(acc_window[:, 1])
                f1_mean = np.mean(acc_window[:, 2])
                f2_mean = np.mean(acc_window[:, 3])

                targets = np.array([
                    np.clip(f0_mean / 500.0, 0, 1),      # Pitch_Shift
                    np.clip(f1_mean / 1000.0, 0, 1),     # Volume_Gain (F1 proxy)
                    np.clip(f2_mean / 2000.0, 0, 1),     # Tempo_Modifier (F2 proxy)
                ], dtype=np.float32)

                X_all.append(features)
                Y_all.append(targets)

            except (IndexError, ValueError):
                continue

    if len(X_all) == 0:
        print("[!] No real data found. Using synthetic data.")
        return _generate_synthetic_data(n_timesteps)

    return np.stack(X_all), np.stack(Y_all)

def _generate_synthetic_data(n_timesteps=32, n_samples=1000):
    """
    Generates physically plausible synthetic training data.
    Uses sinusoidal physiological patterns with correlated targets.
    """
    X = np.zeros((n_samples, n_timesteps, 4), dtype=np.float32)
    Y = np.zeros((n_samples, 3), dtype=np.float32)
    t = np.linspace(0, 1, n_timesteps)

    for i in range(n_samples):
        f0_base = np.random.uniform(0.1, 0.9)
        rms_base = np.random.uniform(0.02, 0.8)

        # Simulate f0 contour with vibrato
        X[i, :, 0] = f0_base + 0.05 * np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2*np.pi))
        # Simulate RMS energy envelope with exponential decay
        X[i, :, 1] = rms_base * np.exp(-np.random.uniform(0.5, 3.0) * t)
        # Simulate MNF and MDF
        X[i, :, 2] = np.random.uniform(0.3, 0.7) + 0.05 * np.random.randn(n_timesteps)
        X[i, :, 3] = np.random.uniform(0.4, 0.8) + 0.05 * np.random.randn(n_timesteps)

        X[i] = np.clip(X[i], 0, 1)

        # Correlated targets
        Y[i, 0] = np.clip(f0_base + np.random.normal(0, 0.05), 0, 1)   # Pitch_Shift
        Y[i, 1] = np.clip(rms_base * 1.2 + np.random.normal(0, 0.05), 0, 1)  # Volume_Gain
        Y[i, 2] = np.clip(np.mean(X[i, :, 3]) + np.random.normal(0, 0.05), 0, 1)  # Tempo

    return X, Y

# ─── INT8 Quantization ─────────────────────────────────────────────────────────

def convert_int8_tflite(model, representative_dataset_fn):
    """
    Full INT8 Post-Training Quantization.
    A representative dataset is mandatory to prevent clipping of microvolt sEMG variances.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32    # Keep float32 for Cortex-M7 FPU
    converter.inference_output_type = tf.float32
    return converter.convert()

def get_representative_dataset(X_calib, n_timesteps=32):
    """
    Returns a generator yielding calibration inputs for INT8 quantization.
    Must represent the full physiological range of real sEMG/PZT signals.
    """
    def representative_data_gen():
        for i in range(min(len(X_calib), 100)):
            sample = X_calib[i:i+1].astype(np.float32)
            yield [sample]
    return representative_data_gen

# ─── C-Byte Array Export ───────────────────────────────────────────────────────

def to_c_array(tflite_bytes, array_name):
    """Python-native equivalent of `xxd -i model.tflite`"""
    hex_vals = [hex(b) for b in tflite_bytes]
    c_str = f"alignas(16) const unsigned char {array_name}[] = {{\n"
    for i in range(0, len(hex_vals), 12):
        c_str += "  " + ", ".join(hex_vals[i:i+12]) + ",\n"
    c_str += "};\n"
    c_str += f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n"
    return c_str

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    N_TIMESTEPS = 32

    print("[+] Loading affective training dataset...")
    X, Y = load_affective_dataset(DATASET_DIR, N_TIMESTEPS)
    print(f"    -> Dataset: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")

    # Train/validation split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    print("[+] Building CNN 2 Affective Modulator architecture...")
    model = build_affective_model(n_timesteps=N_TIMESTEPS)
    model.summary()

    # ── Training ────────────────────────────────────────────────────────────────
    print("[+] Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)
    ]
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # ── Validation Metrics ───────────────────────────────────────────────────────
    print("\n[!] Running Prosodic Output Validation...")
    predictions = model.predict(X_val, verbose=0)
    mae_pitch  = np.mean(np.abs(Y_val[:, 0] - predictions[:, 0]))
    mae_volume = np.mean(np.abs(Y_val[:, 1] - predictions[:, 1]))
    mae_tempo  = np.mean(np.abs(Y_val[:, 2] - predictions[:, 2]))
    print(f"    -> Pitch_Shift  MAE: {mae_pitch:.4f}")
    print(f"    -> Volume_Gain  MAE: {mae_volume:.4f}")
    print(f"    -> Tempo_Modifier MAE: {mae_tempo:.4f}")

    # ── INT8 Quantization ────────────────────────────────────────────────────────
    print("\n[+] Converting to INT8 TFLite flatbuffer...")
    representative_dataset_fn = get_representative_dataset(X_train)
    tflite_int8 = convert_int8_tflite(model, representative_dataset_fn)

    # Save raw .tflite binary
    tflite_path = "affective_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_int8)
    print(f"    -> Saved: {tflite_path} ({len(tflite_int8)} bytes)")

    # ── C-Byte Array Export ──────────────────────────────────────────────────────
    print("\n[+] Exporting C-byte array to dual_models.h...")
    affective_c = to_c_array(tflite_int8, "affective_model_tflite")

    # Read existing header if it exists (to preserve phonetic model bytes)
    existing_phonetic = ""
    if os.path.exists(OUTPUT_HEADER):
        with open(OUTPUT_HEADER, "r") as f:
            content = f.read()
        phonetic_start = content.find("alignas(16) const unsigned char phonetic_model_tflite")
        phonetic_end = content.find("alignas(16) const unsigned char affective_model_tflite")
        if phonetic_start != -1 and phonetic_end != -1:
            existing_phonetic = content[phonetic_start:phonetic_end].strip()
        elif phonetic_start != -1:
            # No affective section yet, grab everything until #endif
            phonetic_end_alt = content.rfind("#endif")
            existing_phonetic = content[phonetic_start:phonetic_end_alt].strip()

    header_content = f"""#ifndef DUAL_MODELS_H
#define DUAL_MODELS_H

// Auto-generated by CADENCE Dual Model Training Pipeline
// Model A (Phonetic):  FLOAT32 TFLite 1D-CNN — 6-DOF sublingual vector
// Model B (Affective): INT8   TFLite 1D-CNN — 3-DOF prosodic modifier

{existing_phonetic if existing_phonetic else '// [PLACEHOLDER] Model A bytes — run train.py to generate'}

{affective_c}
#endif
"""

    with open(OUTPUT_HEADER, "w") as f:
        f.write(header_content)
    print(f"[SUCCESS] affective_model_tflite[] written to {OUTPUT_HEADER}")

if __name__ == "__main__":
    main()
