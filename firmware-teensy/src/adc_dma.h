#pragma once
#include <Arduino.h>
#include <DMAChannel.h>

/**
 * @brief High-Speed Dual-ADC Direct Memory Access (DMA) Controller.
 * 
 * Bypasses blocking analogRead() calls. Utilizes the Teensy 4.1 DMA controller 
 * and Programmable Delay Block (PDB) to sample A0, A1, A2, and A3 at exactly 2000 Hz.
 * ADC1 handles A0, A1 (sEMG). ADC2 handles A2, A3 (PZT/EDA).
 */
class AdcDmaController {
public:
    AdcDmaController();

    /**
     * @brief Initializes the ADC hardware, DMA channels, and PDB hardware timer.
     * @param targetBuffer Pointer to the circular buffer (4x512)
     * @param numChannels Number of ADC channels (4)
     * @param bufferDepth Number of samples per channel (512)
     */
    void begin(volatile uint16_t* targetBuffer, uint16_t numChannels, uint16_t bufferDepth);

    bool isFrameReady();
    void clearFrameReady();

private:
    DMAChannel dma_adc1;
    DMAChannel dma_adc2;
    volatile bool frame_ready_flag = false;

    // Static ISR for DMA completion
    static void dma_isr();
    
    // Singleton pointer for ISR routing
    static AdcDmaController* instance;
};
