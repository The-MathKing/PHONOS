# CADENCE: Subvocal Synthesis Interface (SSI)

**Project CADENCE** is an advanced, patient-specific sEMG/PZT neuro-prosthetic designed to restore expressive, real-time vocalization to non-verbal patients. By leveraging multi-channel physiological sensors and a bifurcated Edge AI architecture, CADENCE translates microscopic neuromuscular intent and affective state directly into synthesized speech.

Target Hardware: **Teensy 4.1 (ARM Cortex-M7 @ 600MHz)**

---

## 🧠 The Bifurcated Emotion Architecture

At the core of CADENCE is a dual-pipeline machine learning and digital signal processing (DSP) architecture operating simultaneously within the microcontroller's RAM:

### 1. Model A: Phonetic Intent (6-DOF)
- **Input:** Submental sEMG (A0, A1) and PZT mechanomyography (A2) isolated via an embedded FastICA matrix.
- **Output:** A 6-Degrees of Freedom array mapping spatial intent: `Pitch`, `Yaw`, `Intensity`, `Formant 1`, `Formant 2`, and `Formant 3`.
- **Audio Routing:** Dynamically modulates the center frequencies of an FPU-accelerated CMSIS-DSP Biquad Cascade to shape a human-like vocal tract resonance.

### 2. Model B: Affective State (2-DOF)
- **Input:** Electrodermal Activity (EDA) tonic and phasic shifts (A3) compared against an auto-calibrated standard deviation baseline.
- **Output:** A 2-DOF emotional state matrix: `Arousal` and `Valence`.
- **Emotion Modulation:**
  - **High Arousal:** Triggers an ultra-fast amplitude envelope attack (2ms) and routes the core oscillator through a steep `AudioEffectWaveshaper` to induce aggressive harmonic saturation.
  - **Low Arousal / Negative Valence:** Triggers a 4-6Hz `AudioSynthWaveform` LFO vibrato on the glottal source, downshifts the fundamental pitch, and tightly narrows the Q-Factor of the vocal formants to simulate vocal tension.

---

## 🚀 Repository Structure & Execution Phases

To ensure stability during physical prototyping, the firmware codebase is strictly divided into three **PlatformIO Environments**. You can compile and flash the codebase depending on which hardware phase you are currently testing.

### Phase A: Digital Proof (`[env:phase_a]`)
*Validates the dual-ML pipeline execution ceilings using mathematical synthesis.*
- **Firmware:** Bypasses the ADC and instead triggers a 2000Hz hardware timer (`IntervalTimer`) to inject synthetic sinusoidal data into the memory buffer.
- **Command:** `pio run -e phase_a --target upload`

### Phase B: Analog Hardware & Real-Time DSP (`[env:phase_b]`)
*Validates physical sensor ingestion, DMA ring buffers, and Signal-to-Noise Ratio (SNR).*
- **Firmware:** Initializes parallel ADC1/ADC2 Direct Memory Access (DMA). Streams raw analog data through the 2x2 FastICA matrix and 20Hz 2nd-order Butterworth high-pass filter. Bypasses ML inference to stream **raw data** over USB for oscilloscope debugging.
- **Command:** `pio run -e phase_b --target upload`

### Phase C: Human-in-the-Loop Integration (`[env:phase_c]`)
*The final, production-ready neuro-prosthetic code.*
- **Firmware:** Unifies Phase B's DMA/DSP pipeline with Phase A's Machine Learning and Audio Synthesis pipeline. Enforces a strict 64-sample audio block size (`-D AUDIO_BLOCK_SAMPLES=64`) to guarantee sub-3ms system latency.
- **Command:** `pio run -e phase_c --target upload`

---

## 🖥️ Web Diagnostic Dashboard

CADENCE features a real-time, browser-based telemetry dashboard utilizing **Three.js** to visualize the patient's internal state via a 3D vocal mesh and an Emotional Particle Aura.

Because the system streams data at high speed via a proprietary binary USB frame (`[0xAA, 32-bytes, 0xBB]`), a local Node.js relay server translates the stream into WebSockets.

### Running the Dashboard
1. Ensure your Teensy 4.1 is plugged in and running either Phase A or Phase C firmware.
2. Navigate to the diagnostic app folder and run the local relay server:
   ```bash
   cd diagnostic-app
   npm install
   node server.js
   ```
3. Open `diagnostic-app/index.html` in your browser.

*Note: If you wish to host the UI remotely on GitHub Pages, refer to the [GITHUB_PAGES_SETUP.md](./GITHUB_PAGES_SETUP.md) guide.*

---

## 🛠 Dependencies
- [PlatformIO](https://platformio.org/)
- Teensyduino Audio Library
- CMSIS-DSP (arm_math.h)
- Node.js & NPM (For diagnostic UI)
