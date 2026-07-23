#pragma once

#include <Arduino.h>
#include <stdint.h>
#include <arm_math.h>

void dsp_fastica_init();

/**
 * @brief Processes the 4-channel ring buffer.
 * Runs a 2x2 FastICA on A0 and A1 to strip common-mode noise.
 * Applies a 20Hz Butterworth High-Pass filter to A0 and A1.
 * Passes A2 and A3 through cleanly.
 *
 * @param raw_in   Input: raw[channel][sample] — uint16_t ADC readings (0–4095)
 * @param sources  Output: sources[channel][sample] — float32 processed signals
 * @param num_ch   Number of channels (4)
 * @param num_samp Samples per channel (512)
 */
void dsp_fastica_process(
    const uint16_t raw_in[4][512],
    float          sources[4][512],
    uint8_t        num_ch,
    uint16_t       num_samp
);
