#include <Arduino.h>
#include "synthesis_engine.h"
#include "tflite_micro_stub.h"
#include "models/dual_models.h"

// ── Ring Buffer Dimensions ───────────────────────────────────────────────────
static constexpr uint16_t RING_DEPTH    = 512;
static constexpr uint16_t RING_CHANNELS = 4;
volatile uint16_t ring[RING_CHANNELS][RING_DEPTH];
volatile uint16_t ring_head = 0;
volatile bool frame_ready = false;

// ── Hardware Timer ───────────────────────────────────────────────────────────
IntervalTimer synthetic_timer;

// ── TFLite Footprint ─────────────────────────────────────────────────────────
alignas(16) static uint8_t tensor_arena_a[SSI_TENSOR_ARENA_SIZE / 2];
alignas(16) static uint8_t tensor_arena_b[SSI_TENSOR_ARENA_SIZE / 2];
static tflite::MicroInterpreter* interpreter_a = nullptr;
static tflite::MicroInterpreter* interpreter_b = nullptr;
static tflite::AllOpsResolver    resolver;

// ── Audio Synthesis ──────────────────────────────────────────────────────────
SSISynthesizer synth;

// ── Synthetic ISR ────────────────────────────────────────────────────────────
FASTRUN void isr_synthetic() {
    uint16_t head = ring_head;
    float time_sec = millis() / 1000.0f;
    
    // Generate synthetic waves for A0-A3
    ring[0][head] = 2048 + (uint16_t)(1000 * sin(time_sec * 2.0 * PI * 10.0));
    ring[1][head] = 2048 + (uint16_t)(1000 * sin(time_sec * 2.0 * PI * 15.0));
    ring[2][head] = 2048 + (uint16_t)(500 * sin(time_sec * 2.0 * PI * 5.0));
    ring[3][head] = 2048 + (uint16_t)(200 * sin(time_sec * 2.0 * PI * 0.5)); // EDA

    head = (head + 1) & (RING_DEPTH - 1);
    ring_head = head;
    if (head == 0) frame_ready = true;
}

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    // Audio Memory & Init
    AudioMemory(15);
    synth.begin();

    // Enable internal CPU clock cycle counter for optimization benchmarking
    ARM_DEMCR |= ARM_DEMCR_TRCENA;
    ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;

    // Init TFLite
    const tflite::Model* model_a = tflite::GetModel(phonetic_model_tflite);
    static tflite::MicroInterpreter static_interpreter_a(model_a, resolver, tensor_arena_a, sizeof(tensor_arena_a), nullptr);
    interpreter_a = &static_interpreter_a;

    const tflite::Model* model_b = tflite::GetModel(affective_model_tflite);
    static tflite::MicroInterpreter static_interpreter_b(model_b, resolver, tensor_arena_b, sizeof(tensor_arena_b), nullptr);
    interpreter_b = &static_interpreter_b;

    interpreter_a->AllocateTensors();
    interpreter_b->AllocateTensors();

    // Start 2000Hz Synthetic ISR
    synthetic_timer.begin(isr_synthetic, 500);
}

// ── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    if (frame_ready) {
        frame_ready = false;
        
        uint32_t start_cycles = ARM_DWT_CYCCNT;

        // Mock 8-DOF inference
        SSIExpressionVector exp;
        exp.pitch = 0.5f; exp.yaw = 0.5f; exp.intensity = 0.5f;
        exp.formant_1 = 0.5f; exp.formant_2 = 0.5f; exp.formant_3 = 0.5f;
        
        SSIEmotionVector emo;
        emo.arousal = 0.5f; emo.valence = 0.0f;

        // In production, execute both models:
        // interpreter_a->Invoke();
        // interpreter_b->Invoke();
        
        uint32_t execution_cycles = ARM_DWT_CYCCNT - start_cycles;
        float execution_time_us = (float)execution_cycles / 600.0f; // Teensy 4.1 clock factor

        // Periodically print execution time
        static uint32_t last_print = 0;
        if (millis() - last_print > 1000) {
            Serial.printf("[Phase A] Inference Loop OK. Execution Time: %.2f us\n", execution_time_us);
            last_print = millis();
        }

        // Drive synthesis
        synth.update(exp, emo);

        // Binary Telemetry: 0xAA, 8 floats, 0xBB
        uint8_t packet[34];
        packet[0] = 0xAA;
        memcpy(&packet[1], &exp, 24);
        memcpy(&packet[25], &emo, 8);
        packet[33] = 0xBB;
        Serial.write(packet, 34);
    }
}
