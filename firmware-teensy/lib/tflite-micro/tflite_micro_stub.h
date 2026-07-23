/**
 * @file    tflite-micro/tflite_micro_stub.h
 * @brief   TensorFlow Lite for Microcontrollers — Dependency Footprint Stub
 * @note    This file is a build system stub that declares the TFLite-Micro
 *          API surface expected by the SSI inference engine.
 *
 *          In production, replace this with the full TFLite-Micro source tree:
 *          https://github.com/tensorflow/tflite-micro
 *
 *          To integrate the real library:
 *          1. Clone tflite-micro repo into firmware-teensy/lib/tflite-micro/
 *          2. Remove this stub file
 *          3. PlatformIO will automatically discover the library via lib_extra_dirs
 *
 * Expected API subset for SSI inference:
 *   - tflite::MicroInterpreter    — Run model inference
 *   - tflite::MicroErrorReporter  — Serial debug output
 *   - tflite::AllOpsResolver      — Registers all built-in ops
 *   - tflite::GetModel()          — Load model from flash-resident flatbuffer
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

// ── TFLite-Micro Stub Namespace ──────────────────────────────────────────────
namespace tflite {

// Tensor types supported by the SSI model
enum TensorType : uint8_t {
    kTfLiteFloat32 = 0,
    kTfLiteInt8    = 1,
    kTfLiteUInt8   = 2,
};

// Stub: Error reporter base class
class ErrorReporter {
public:
    virtual int Report(const char* format, ...) = 0;
    virtual ~ErrorReporter() = default;
};

// Stub: Serial error reporter (writes to Arduino Serial)
class MicroErrorReporter : public ErrorReporter {
public:
    int Report(const char* /*format*/, ...) override { return 0; }
};

// Stub: Flat model representation
struct Model {
    uint32_t version;
};

// Stub: Get model from a flatbuffer byte array in flash
inline const Model* GetModel(const void* /*data*/) {
    return nullptr; // Stub: real impl reads from PROGMEM flatbuffer
}

// Stub: All ops resolver — registers Conv1D, Dense, BatchNorm, etc.
class AllOpsResolver {
public:
    AllOpsResolver() = default;
};

// Stub: Tensor
struct TfLiteTensor {
    TensorType type;
    float*     data_float;
    size_t     bytes;
    int        dims[4];
};

// Stub: Interpreter — runs one inference pass
class MicroInterpreter {
public:
    MicroInterpreter(
        const Model*       /*model*/,
        const AllOpsResolver& /*resolver*/,
        uint8_t*           /*tensor_arena*/,
        size_t             /*arena_size*/,
        ErrorReporter*     /*error_reporter*/
    ) {}

    // Returns kTfLiteOk (0) on success
    int AllocateTensors() { return 0; }
    int Invoke()          { return 0; }

    TfLiteTensor* input(int /*idx*/)  { return nullptr; }
    TfLiteTensor* output(int /*idx*/) { return nullptr; }
};

} // namespace tflite

// ── SSI Inference Output: 6-DOF Expression Vector ───────────────────────────
/**
 * @brief Output structure for the SSI expression inference engine.
 *
 * Maps to the 1D-CNN regression output: a 6-dimensional continuous vector
 * representing the full vocal tract gesture state:
 *
 *   [0] pitch     — Fundamental frequency deviation (normalized, −1..+1)
 *   [1] yaw       — Lateral jaw movement proxy    (normalized, −1..+1)
 *   [2] intensity — Subvocal effort magnitude     (normalized,  0..+1)
 *   [3] formant_1 — First  formant (F1) axis      (normalized, −1..+1)
 *   [4] formant_2 — Second formant (F2) axis      (normalized, −1..+1)
 *   [5] formant_3 — Third  formant (F3) axis      (normalized, −1..+1)
 */
struct SSIExpressionVector {
    float pitch;
    float yaw;
    float intensity;
    float formant_1;
    float formant_2;
    float formant_3;
};

// Model inference tensor arena size (bytes reserved in DTCM RAM)
static constexpr size_t SSI_TENSOR_ARENA_SIZE = 64 * 1024; // 64 KB
