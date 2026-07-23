#include "adc_dma.h"

AdcDmaController* AdcDmaController::instance = nullptr;

AdcDmaController::AdcDmaController() {
    instance = this;
}

void AdcDmaController::begin(volatile uint16_t* targetBuffer, uint16_t numChannels, uint16_t bufferDepth) {
    // Note: This is the architectural framework for i.MX RT1062 Dual ADC DMA.
    // ADC1 handles A0, A1 (sEMG). ADC2 handles A2, A3 (PZT/EDA).
    
    // 1. Initialize ADC1 DMA
    dma_adc1.begin(true);
    dma_adc1.source((volatile uint16_t&)ADC1_R0);
    // Writes to channels 0 and 1 of the ring buffer
    dma_adc1.destinationBuffer((volatile uint16_t*)targetBuffer, bufferDepth * 2 * sizeof(uint16_t));
    dma_adc1.triggerAtHardwareEvent(DMAMUX_SOURCE_ADC1);
    
    // 2. Initialize ADC2 DMA
    dma_adc2.begin(true);
    dma_adc2.source((volatile uint16_t&)ADC2_R0);
    // Offset target buffer to write to channels 2 and 3
    dma_adc2.destinationBuffer((volatile uint16_t*)(targetBuffer + (bufferDepth * 2)), bufferDepth * 2 * sizeof(uint16_t));
    dma_adc2.triggerAtHardwareEvent(DMAMUX_SOURCE_ADC2);
    
    // 3. Attach ISR to ADC1 (Assuming synchronous triggering via PDB)
    dma_adc1.attachInterrupt(dma_isr);
    dma_adc1.interruptAtCompletion();
    
    // 4. Enable DMA
    dma_adc1.enable();
    dma_adc2.enable();
}

bool AdcDmaController::isFrameReady() {
    return frame_ready_flag;
}

void AdcDmaController::clearFrameReady() {
    frame_ready_flag = false;
}

void AdcDmaController::dma_isr() {
    if (instance) {
        instance->frame_ready_flag = true;
    }
    // Clear DMA interrupt flag
    instance->dma_adc1.clearInterrupt();
}
