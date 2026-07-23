/**
 * @file    main.cpp
 * @brief   SSI Neuro-Prosthetic — Real-Time Telemetry Layer
 * @target  Teensy 4.1 (NXP IMXRT1062, ARM Cortex-M7 @ 600 MHz)
 *
 * Implements an interrupt-driven, 2000 Hz multi-channel ADC sampling loop
 * for subvocal gesture capture:
 *   CH0 — Left  submental sEMG  (surface electromyography)
 *   CH1 — Right submental sEMG
 *   CH2 — PZT film MMG          (piezo mechanomyography vibration)
 *
 * Data is stored in a 3-channel × 512-sample circular ring buffer.
 * When the buffer completes one full revolution, a calibration frame is
 * serialized to the Teensy 4.1 onboard SD card in a self-describing
 * binary format: [magic][timestamp_ms][sample_count][raw_data].
 *
 * DSP (FastICA source separation) is invoked in main loop on complete frames.
 */

#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#include "dsp_processing.h"

// ── Pin Definitions ──────────────────────────────────────────────────────────
static constexpr uint8_t PIN_SEMG_LEFT  = A0;   ///< Left  submental sEMG input
static constexpr uint8_t PIN_SEMG_RIGHT = A1;   ///< Right submental sEMG input
static constexpr uint8_t PIN_MMG_PZT    = A2;   ///< PZT film MMG input
static constexpr uint8_t SD_CS_PIN      = BUILTIN_SDCARD; ///< Teensy 4.1 built-in SDIO

// ── Sampling Parameters ──────────────────────────────────────────────────────
static constexpr uint32_t SAMPLE_RATE_HZ   = 2000;
static constexpr uint32_t INTERVAL_US      = 1000000UL / SAMPLE_RATE_HZ; // 500 µs
static constexpr uint16_t RING_CHANNELS    = 3;
static constexpr uint16_t RING_DEPTH       = 512;

// ── Circular Ring Buffer ─────────────────────────────────────────────────────
// Layout: ring[channel][sample_index]
// Channel 0 = L-sEMG, 1 = R-sEMG, 2 = MMG
volatile uint16_t ring[RING_CHANNELS][RING_DEPTH];
volatile uint16_t ring_head  = 0;   ///< Write pointer (0..RING_DEPTH-1)
volatile bool     frame_ready = false; ///< Set by ISR when buffer wraps

// ── Calibration File State ───────────────────────────────────────────────────
static File     cal_file;
static uint32_t frame_count = 0;

// ── SD Binary Frame Header ───────────────────────────────────────────────────
// Magic identifier + metadata for self-describing binary blobs
#pragma pack(push, 1)
struct CalibrationFrameHeader {
    uint8_t  magic[4];       ///< {'S','S','I','0'}
    uint32_t timestamp_ms;   ///< millis() at frame capture
    uint32_t sample_count;   ///< total samples since boot (cumulative)
    uint16_t num_channels;   ///< always RING_CHANNELS (3)
    uint16_t samples_per_ch; ///< always RING_DEPTH (512)
};
#pragma pack(pop)

// ── Hardware Timer ───────────────────────────────────────────────────────────
IntervalTimer sample_timer;

// ── ISR: ADC Sampling ────────────────────────────────────────────────────────
/**
 * @brief Hardware timer ISR — fires every 500 µs (2000 Hz).
 *        Reads all three ADC channels into the ring buffer.
 *        Sets frame_ready flag on buffer wrap (every 512 samples).
 *
 * Teensy 4.1 ADC is 12-bit by default; readings range 0–4095.
 * Stored directly as uint16_t (upper 4 bits unused, valid for microvolt
 * calibration math in the ML pipeline).
 */
FASTRUN void isr_sample() {
    uint16_t head = ring_head;

    // Sample all three channels sequentially (< 1 µs total on M7 @ 600 MHz)
    ring[0][head] = static_cast<uint16_t>(analogRead(PIN_SEMG_LEFT));
    ring[1][head] = static_cast<uint16_t>(analogRead(PIN_SEMG_RIGHT));
    ring[2][head] = static_cast<uint16_t>(analogRead(PIN_MMG_PZT));

    // Advance write pointer with power-of-2 wrap (no division)
    head = (head + 1) & (RING_DEPTH - 1);
    ring_head = head;

    // Signal main loop when the buffer has completed one full revolution
    if (head == 0) {
        frame_ready = true;
    }
}

// ── SD Card Initialization ───────────────────────────────────────────────────
/**
 * @brief Opens (or creates) the patient calibration log on the SD card.
 *        File is kept open for the session to minimize latency per write.
 * @return true on success, false if SD init or file open fails.
 */
static bool init_sd_logging() {
    if (!SD.begin(SD_CS_PIN)) {
        Serial.println("[SSI] ERROR: SD card mount failed.");
        return false;
    }

    // Generate a session-unique filename using millis()
    char filename[32];
    snprintf(filename, sizeof(filename), "CAL_%010lu.ssi", millis());

    cal_file = SD.open(filename, FILE_WRITE);
    if (!cal_file) {
        Serial.print("[SSI] ERROR: Cannot open calibration file: ");
        Serial.println(filename);
        return false;
    }

    Serial.print("[SSI] Calibration log: ");
    Serial.println(filename);
    return true;
}

// ── Frame Serialization ──────────────────────────────────────────────────────
/**
 * @brief Serializes one complete ring buffer snapshot to the SD card.
 *
 * Binary frame layout:
 *   [CalibrationFrameHeader: 16 bytes]
 *   [uint16_t[3][512]: 3072 bytes — channel-major order]
 *
 * Total per frame: 3088 bytes @ 2000 Hz / 512 samples → ~3.9 fps frame rate
 *
 * Data validation:
 *   - Magic byte check ensures header integrity before payload write.
 *   - Interrupt guard (noInterrupts/interrupts) prevents ring_head mutation
 *     during the snapshot copy.
 */
static void serialize_frame_to_sd() {
    // ── Snapshot ring buffer into a local copy (interrupt-safe) ──
    uint16_t snapshot[RING_CHANNELS][RING_DEPTH];

    noInterrupts();
    memcpy(snapshot, (const void*)ring, sizeof(snapshot));
    uint32_t capture_ms = millis();
    interrupts();

    // ── Validation: sanity-check value ranges before writing ──
    // sEMG and MMG ADC values must be within 12-bit range [0, 4095]
    for (uint8_t ch = 0; ch < RING_CHANNELS; ch++) {
        for (uint16_t i = 0; i < RING_DEPTH; i++) {
            if (snapshot[ch][i] > 4095) {
                Serial.println("[SSI] WARN: ADC overflow detected — clamping to 4095");
                snapshot[ch][i] = 4095;
            }
        }
    }

    // ── Build header ──
    CalibrationFrameHeader hdr;
    hdr.magic[0]       = 'S'; hdr.magic[1] = 'S';
    hdr.magic[2]       = 'I'; hdr.magic[3] = '0';
    hdr.timestamp_ms   = capture_ms;
    hdr.sample_count   = (frame_count + 1) * RING_DEPTH;
    hdr.num_channels   = RING_CHANNELS;
    hdr.samples_per_ch = RING_DEPTH;

    // ── Write header + payload to SD ──
    if (!cal_file) {
        Serial.println("[SSI] ERROR: cal_file not open — skipping frame write.");
        return;
    }

    // Magic byte guard before write
    if (hdr.magic[0] != 'S' || hdr.magic[3] != '0') {
        Serial.println("[SSI] ERROR: Frame header validation failed — aborting write.");
        return;
    }

    cal_file.write(reinterpret_cast<const uint8_t*>(&hdr),    sizeof(hdr));
    cal_file.write(reinterpret_cast<const uint8_t*>(snapshot), sizeof(snapshot));
    cal_file.flush(); // Ensure durability after each frame

    frame_count++;
    Serial.printf("[SSI] Frame %lu written | t=%lu ms | %u bytes\n",
                  frame_count, capture_ms,
                  (unsigned)(sizeof(hdr) + sizeof(snapshot)));
}

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000); // Wait for USB serial (max 3 s)

    Serial.println("\n[SSI] Subvocal Synthesis Interface — Firmware v0.1.0");
    Serial.println("[SSI] Target: Teensy 4.1 | Rate: 2000 Hz | Channels: 3");

    // Configure ADC: 12-bit resolution, hardware averaging for noise reduction
    analogReadResolution(12);
    analogReadAveraging(4); // 4-sample HW average: ~20 µs per channel

    // Initialize SD card for calibration logging
    if (!init_sd_logging()) {
        Serial.println("[SSI] WARN: SD logging disabled — running without persistence.");
    }

    // Start hardware timer at exactly 2000 Hz (500 µs period)
    if (!sample_timer.begin(isr_sample, INTERVAL_US)) {
        Serial.println("[SSI] FATAL: IntervalTimer allocation failed.");
        while (true); // Halt on timer failure
    }

    Serial.println("[SSI] Sampling started at 2000 Hz.");
}

// ── Main Loop ────────────────────────────────────────────────────────────────
void loop() {
    // Poll for completed ring buffer frame
    if (frame_ready) {
        // Clear flag before processing (safe: main loop only writer of this flag)
        frame_ready = false;

        // Serialize raw calibration frame to SD
        serialize_frame_to_sd();

        // Apply FastICA source separation to isolate vocal intent
        // (Takes a local snapshot internally for ISR safety)
        noInterrupts();
        uint16_t local_ring[RING_CHANNELS][RING_DEPTH];
        memcpy(local_ring, (const void*)ring, sizeof(local_ring));
        interrupts();

        float separated[RING_CHANNELS][RING_DEPTH];
        dsp_fastica_process(
            local_ring,
            separated,
            RING_CHANNELS,
            RING_DEPTH
        );

        // TODO: Route separated[] to inference engine (TFLite-Micro)
    }
}
