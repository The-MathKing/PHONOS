#include <Arduino.h>
#include <arm_math.h>
#include "dsp_processing.h"
#include "synthesis_engine.h"
#include "tflite_micro_stub.h"
#include "adc_dma.h"
#include "affective_dsp.h"

// ── Ring Buffer Dimensions ───────────────────────────────────────────────────
static constexpr uint16_t RING_DEPTH    = 512;
static constexpr uint16_t RING_CHANNELS = 4;
volatile uint16_t ring[RING_CHANNELS][RING_DEPTH];

AdcDmaController adc_dma;
SSISynthesizer synth;

// TFLite memory footprint (statically allocated in DTCM RAM)
alignas(16) static uint8_t tensor_arena_a[SSI_TENSOR_ARENA_SIZE / 2];
alignas(16) static uint8_t tensor_arena_b[SSI_TENSOR_ARENA_SIZE / 2];
static tflite::MicroInterpreter* interpreter_a = nullptr; 
static tflite::MicroInterpreter* interpreter_b = nullptr; 
static tflite::AllOpsResolver    resolver;

// ── Auto-Calibration ─────────────────────────────────────────────────────────
bool calibration_complete = false;
uint32_t calib_start_time = 0;
float eda_sum = 0.0f;
float eda_sq_sum = 0.0f;
uint32_t eda_samples = 0;
float eda_baseline_stddev = 0.0f;
float eda_baseline_mean = 0.0f;

float semg_sum[2] = {0.0f, 0.0f};
float semg_sq_sum[2] = {0.0f, 0.0f};
float semg_baseline_mean[2] = {0.0f, 0.0f};
float semg_baseline_stddev[2] = {0.0f, 0.0f};

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    AudioMemory(15);
    synth.begin();
    dsp_fastica_init();
    affective_dsp_init();

    // Init TFLite
    const tflite::Model* model = tflite::GetModel(nullptr);
    static tflite::MicroInterpreter static_interpreter_a(model, resolver, tensor_arena_a, sizeof(tensor_arena_a), nullptr);
    interpreter_a = &static_interpreter_a;
    static tflite::MicroInterpreter static_interpreter_b(model, resolver, tensor_arena_b, sizeof(tensor_arena_b), nullptr);
    interpreter_b = &static_interpreter_b;
    interpreter_a->AllocateTensors();
    interpreter_b->AllocateTensors();

    // Enable internal CPU clock cycle counter for latency profiling
    ARM_DEMCR |= ARM_DEMCR_TRCENA;
    ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;

    adc_dma.begin((volatile uint16_t*)ring, RING_CHANNELS, RING_DEPTH);
    calib_start_time = millis();
}

void loop() {
    if (adc_dma.isFrameReady()) {
        adc_dma.clearFrameReady();

        noInterrupts();
        uint16_t local_ring[RING_CHANNELS][RING_DEPTH];
        memcpy(local_ring, (const void*)ring, sizeof(local_ring));
        interrupts();

        float separated[RING_CHANNELS][RING_DEPTH];
        dsp_fastica_process(local_ring, separated, RING_CHANNELS, RING_DEPTH);

        // 1. Auto-Calibration (First 5 seconds)
        if (!calibration_complete) {
            for (int i=0; i<RING_DEPTH; i++) {
                float v_eda = separated[3][i];
                float v_a0 = separated[0][i];
                float v_a1 = separated[1][i];
                
                eda_sum += v_eda;
                eda_sq_sum += (v_eda * v_eda);
                
                semg_sum[0] += v_a0;
                semg_sq_sum[0] += (v_a0 * v_a0);
                
                semg_sum[1] += v_a1;
                semg_sq_sum[1] += (v_a1 * v_a1);
                
                eda_samples++;
            }
            if (millis() - calib_start_time > 5000) {
                eda_baseline_mean = eda_sum / eda_samples;
                eda_baseline_stddev = sqrt((eda_sq_sum / eda_samples) - (eda_baseline_mean * eda_baseline_mean));
                
                for (int ch=0; ch<2; ch++) {
                    semg_baseline_mean[ch] = semg_sum[ch] / eda_samples;
                    semg_baseline_stddev[ch] = sqrt((semg_sq_sum[ch] / eda_samples) - (semg_baseline_mean[ch] * semg_baseline_mean[ch]));
                }
                
                calibration_complete = true;
                Serial.println("[SSI] Edge Calibration Complete. Normalizing to [0.0, 1.0]");
            }
            return; // Skip inference until calibrated
        }

        // 2. Real-Time Tensor Normalization [0.0, 1.0]
        float normalized[RING_CHANNELS][RING_DEPTH];
        for (int i = 0; i < RING_DEPTH; i++) {
            // Normalize sEMG using baseline mean and 3 standard deviations (99.7% capture)
            for (int ch = 0; ch < 2; ch++) {
                float v = separated[ch][i];
                float range = semg_baseline_stddev[ch] * 3.0f + 1e-6f;
                float norm_v = (v - semg_baseline_mean[ch] + range) / (2.0f * range);
                normalized[ch][i] = constrain(norm_v, 0.0f, 1.0f);
            }
            // PZT is zero-mean acoustic transient, bypass normalization for 1D-CNN handling
            normalized[2][i] = separated[2][i];
            
            // EDA is handled in emotion mapping
            normalized[3][i] = separated[3][i];
        }

        // 3a. CNN 2: CMSIS-DSP Affective Feature Extraction (Sub-budget: <150us)
        uint32_t affective_start = ARM_DWT_CYCCNT;

        // Build 256-sample analysis windows from the latest RING_DEPTH=512 frame
        // We use the latter 256 samples (most recent) to minimize analysis latency
        float pzt_window[AFFECTIVE_BUFFER_SIZE];
        float semg_window[AFFECTIVE_BUFFER_SIZE];
        const uint32_t WINDOW_OFFSET = RING_DEPTH - AFFECTIVE_BUFFER_SIZE;
        for (uint32_t i = 0; i < AFFECTIVE_BUFFER_SIZE; i++) {
            pzt_window[i]  = normalized[2][WINDOW_OFFSET + i]; // PZT on channel 2
            semg_window[i] = normalized[0][WINDOW_OFFSET + i]; // Primary sEMG on channel 0
        }

        AffectiveFeatureVector affective_features;
        affective_dsp_extract(pzt_window, semg_window, &affective_features);

        uint32_t affective_cycles = ARM_DWT_CYCCNT - affective_start;
        float affective_us = (float)affective_cycles / 600.0f;
        static float max_affective_us = 0;
        if (affective_us > max_affective_us) max_affective_us = affective_us;

        // 3. Inference & Synthesis (Profiled)
        uint32_t start_cycles = ARM_DWT_CYCCNT;
        
        tflite::TfLiteTensor* input = interpreter_a->input(0);
        if (input != nullptr) memcpy(input->data_float, normalized, sizeof(normalized));

        // Mock inference results (Replace with interpreter_a->Invoke() and interpreter_b->Invoke())
        SSIExpressionVector exp_vector;
        exp_vector.pitch     = abs(separated[0][0]) * 0.01f; 
        exp_vector.yaw       = abs(separated[1][0]) * 0.01f; 
        exp_vector.intensity = abs(separated[2][0]) * 0.01f;
        exp_vector.formant_1 = abs(separated[0][1]) * 0.01f;
        exp_vector.formant_2 = abs(separated[1][1]) * 0.01f;
        exp_vector.formant_3 = abs(separated[2][1]) * 0.01f;

        SSIEmotionVector emo_vector;
        emo_vector.arousal = 0.5f; // Safe defaults
        emo_vector.valence = 0.0f;
        
        // 450us execution ceiling bypass for Model B
        uint32_t current_cycles = ARM_DWT_CYCCNT - start_cycles;
        if (current_cycles < (450 * 600)) {
            // ── CNN 2: Affective Modulator Inference ──────────────────────────
            // Feed the 4-DOF physiological feature vector to TFLite Model B.
            // Features: [f0, rms_energy, mnf, mdf] — all in [0.0, 1.0]
            tflite::TfLiteTensor* model_b_input = interpreter_b->input(0);
            if (model_b_input != nullptr) {
                model_b_input->data_float[0] = affective_features.f0;
                model_b_input->data_float[1] = affective_features.rms_energy;
                model_b_input->data_float[2] = affective_features.mnf;
                model_b_input->data_float[3] = affective_features.mdf;
            }
            // In production: interpreter_b->Invoke();
            // For now, derive emotion from CMSIS-DSP features directly:
            // High f0 + high RMS = high arousal; low MDF = muscular fatigue (low valence)
            emo_vector.arousal = constrain((affective_features.f0 * 0.5f + affective_features.rms_energy * 0.5f), 0.0f, 1.0f);
            emo_vector.valence = constrain((affective_features.mdf - 0.5f) * 2.0f, -1.0f, 1.0f);
        } else {
            Serial.println("[!] WARNING: 450us ceiling exceeded. Bypassing Model B.");
        }

        // Audio Synthesis
        synth.update(exp_vector, emo_vector);
        
        uint32_t execution_cycles = ARM_DWT_CYCCNT - start_cycles;
        float execution_time_us = (float)execution_cycles / 600.0f; // Teensy 4.1 clock factor

        // Periodically print max execution time
        static float max_time_us = 0;
        if (execution_time_us > max_time_us) max_time_us = execution_time_us;
        static uint32_t last_print = 0;
        if (millis() - last_print > 1000) {
            Serial.printf("[Phase C] System Latency: %.2f us (Limit: 500 us). Affective DSP: %.2f us (Limit: 150 us)\n",
                          max_time_us, max_affective_us);
            max_time_us = 0;
            max_affective_us = 0;
            last_print = millis();
        }

        // 4. Telemetry: 0xAA, 8 floats, 0xBB
        uint8_t packet[34];
        packet[0] = 0xAA;
        memcpy(&packet[1], &exp_vector, 24);
        memcpy(&packet[25], &emo_vector, 8);
        packet[33] = 0xBB;
        Serial.write(packet, 34);
    }
}
