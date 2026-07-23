#include <Arduino.h>
#include <arm_math.h>
#include "dsp_processing.h"
#include "synthesis_engine.h"
#include "tflite_micro_stub.h"
#include "adc_dma.h"

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

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    AudioMemory(15);
    synth.begin();
    dsp_fastica_init();

    // Init TFLite
    const tflite::Model* model = tflite::GetModel(nullptr);
    static tflite::MicroInterpreter static_interpreter_a(model, resolver, tensor_arena_a, sizeof(tensor_arena_a), nullptr);
    interpreter_a = &static_interpreter_a;
    static tflite::MicroInterpreter static_interpreter_b(model, resolver, tensor_arena_b, sizeof(tensor_arena_b), nullptr);
    interpreter_b = &static_interpreter_b;
    interpreter_a->AllocateTensors();
    interpreter_b->AllocateTensors();

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

        // 1. Auto-Calibration (First 3 seconds)
        if (!calibration_complete) {
            for (int i=0; i<RING_DEPTH; i++) {
                float v = separated[3][i]; // EDA
                eda_sum += v;
                eda_sq_sum += (v * v);
                eda_samples++;
            }
            if (millis() - calib_start_time > 3000) {
                eda_baseline_mean = eda_sum / eda_samples;
                eda_baseline_stddev = sqrt((eda_sq_sum / eda_samples) - (eda_baseline_mean * eda_baseline_mean));
                calibration_complete = true;
                // Optional: Print calibration results
            }
            return; // Skip inference until calibrated
        }

        // 2. Inference
        tflite::TfLiteTensor* input = interpreter_a->input(0);
        if (input != nullptr) memcpy(input->data_float, separated, sizeof(separated));

        // Mock inference results
        SSIExpressionVector exp_vector;
        exp_vector.pitch     = abs(separated[0][0]) * 0.01f; 
        exp_vector.yaw       = abs(separated[1][0]) * 0.01f; 
        exp_vector.intensity = abs(separated[2][0]) * 0.01f;
        exp_vector.formant_1 = abs(separated[0][1]) * 0.01f;
        exp_vector.formant_2 = abs(separated[1][1]) * 0.01f;
        exp_vector.formant_3 = abs(separated[2][1]) * 0.01f;

        SSIEmotionVector emo_vector;
        // Map EDA standard deviation against baseline
        float eda_current_mean = 0.0f;
        arm_mean_f32(separated[3], RING_DEPTH, &eda_current_mean);
        float delta = abs(eda_current_mean - eda_baseline_mean);
        
        emo_vector.arousal = constrain(delta / (eda_baseline_stddev * 3.0f + 1e-6f), 0.0f, 1.0f);
        emo_vector.valence = (emo_vector.arousal * 2.0f) - 1.0f; 

        // 3. Audio Synthesis
        synth.update(exp_vector, emo_vector);

        // 4. Telemetry: 0xAA, 8 floats, 0xBB
        uint8_t packet[34];
        packet[0] = 0xAA;
        memcpy(&packet[1], &exp_vector, 24);
        memcpy(&packet[25], &emo_vector, 8);
        packet[33] = 0xBB;
        Serial.write(packet, 34);
    }
}
