#include <Arduino.h>
#include "adc_dma.h"
#include "dsp_processing.h"

// ── Ring Buffer Dimensions ───────────────────────────────────────────────────
static constexpr uint16_t RING_DEPTH    = 512;
static constexpr uint16_t RING_CHANNELS = 4;
volatile uint16_t ring[RING_CHANNELS][RING_DEPTH];

AdcDmaController adc_dma;

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    // Init DSP logic
    dsp_fastica_init();

    // Start background DMA ADC sampling at 2000 Hz
    adc_dma.begin((volatile uint16_t*)ring, RING_CHANNELS, RING_DEPTH);
}

// ── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    if (adc_dma.isFrameReady()) {
        adc_dma.clearFrameReady();

        noInterrupts();
        uint16_t local_ring[RING_CHANNELS][RING_DEPTH];
        memcpy(local_ring, (const void*)ring, sizeof(local_ring));
        interrupts();

        float separated[RING_CHANNELS][RING_DEPTH];
        dsp_fastica_process(local_ring, separated, RING_CHANNELS, RING_DEPTH);

        // Phase B Raw Telemetry Override
        // Transmit raw isolated streams to PC for digital storage oscilloscope
        // Send first sample as a representative of the frame (for high speed plotting)
        float raw_packet[4];
        raw_packet[0] = separated[0][0]; // L-sEMG Clean
        raw_packet[1] = separated[1][0]; // R-sEMG Clean
        raw_packet[2] = separated[2][0]; // PZT
        raw_packet[3] = separated[3][0]; // EDA

        uint8_t packet[18];
        packet[0] = 0xAA;
        memcpy(&packet[1], raw_packet, 16);
        packet[17] = 0xBB;
        Serial.write(packet, 18);
    }
}
