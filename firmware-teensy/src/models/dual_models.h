#ifndef DUAL_MODELS_H
#define DUAL_MODELS_H

// Model A: Phonetic Regression Model
// Expected to be a quantized 1D-CNN.
// Replace with `xxd -i phonetic.tflite > dual_models.h` later.
alignas(16) const unsigned char phonetic_model_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, // Stub header bytes
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
const unsigned int phonetic_model_tflite_len = 32768; // Stub length

// Model B: Affective State Model
// Expected to be a smaller dense model mapping EDA to arousal/valence.
alignas(16) const unsigned char affective_model_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, // Stub header bytes
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
const unsigned int affective_model_tflite_len = 16384; // Stub length

#endif
