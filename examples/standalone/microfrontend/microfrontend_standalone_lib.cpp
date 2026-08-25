/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>

// extern "C" {
// #include "uart.h"
// }

#ifndef TARGETLIB_RISCV_TIME_H
#define TARGETLIB_RISCV_TIME_H

/**
 * @brief Returns the number of clock cycles executed by the processor.
 * Overflows after 42.9 seconds on a 100 MIPS and with CPI = 1 (OVPsim).
 */
static inline uint32_t rdcycle(void) {
#if defined(__riscv) || defined(__riscv__)
  uint32_t cycles;
  __asm__ volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
#else
  return 0;
#endif
}

/**
 * @brief Returns the full 64bit register cycle register, which holds the
 * number of clock cycles executed by the processor.
 */
static inline uint64_t rdcycle64()
{
#if defined(__riscv) || defined(__riscv__)
#if __riscv_xlen == 32
    uint32_t cycles;
    uint32_t cyclesh1;
    uint32_t cyclesh2;

    /* Reads are not atomic. So ensure, that we are never reading inconsistent
     * values from the 64bit hardware register. */
    do
    {
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh1));
        __asm__ volatile("rdcycle %0" : "=r"(cycles));
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh2));
    } while (cyclesh1 != cyclesh2);

    return (((uint64_t)cyclesh1) << 32) | cycles;
#else
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#endif
#else
    return 0;
#endif
}

/* How many time cycles (rdtime) per second (OVPsim and Spike). */
#define RDTIME_PER_SECOND 1000000UL

/**
 * @brief Returns wall-clock real time that has passed from an arbitrary start time in the past.
 */
static inline uint32_t rdtime(void) {
#if defined(__riscv) || defined(__riscv__)
  uint32_t time;
  __asm__ volatile("rdtime %0" : "=r"(time));
  return time;
#else
  return 0;
#endif
}

/**
 * @brief Returns the number of instructions retired by the processor.
 */
static inline uint32_t rdinstret(void) {
#if defined(__riscv) || defined(__riscv__)
  uint32_t instret;
  __asm__ volatile("rdinstret %0" : "=r"(instret));
  return instret;
#else
  return 0;
#endif
}

/**
 * @brief Returns the full 64bit register cycle register, which holds the
 * number of instructions retired by the processor.
 */
static inline uint64_t rdinstret64()
{
#if defined(__riscv) || defined(__riscv__)
#if __riscv_xlen == 32
    uint32_t instret;
    uint32_t instreth1;
    uint32_t instreth2;

    /* Reads are not atomic. So ensure, that we are never reading inconsistent
     * values from the 64bit hardware register. */
    do
    {
        __asm__ volatile("rdinstreth %0" : "=r"(instreth1));
        __asm__ volatile("rdinstret %0" : "=r"(instret));
        __asm__ volatile("rdinstreth %0" : "=r"(instreth2));
    } while (instreth1 != instreth2);

    return (((uint64_t)instreth1) << 32) | instret;
#else
    uint64_t instret;
    __asm__ volatile("rdinstret %0" : "=r"(instret));
    return instret;
#endif
#else
    return 0;
#endif
}

#endif  // TARGETLIB_RISCV_TIME_H

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FFT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FFT_H_

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct complex_int16_t {
  int16_t real;
  int16_t imag;
};

struct FftState {
  int16_t* input;
  struct complex_int16_t* output;
  size_t fft_size;
  size_t input_size;
  void* scratch;
  size_t scratch_size;
};

void FftCompute(struct FftState* state, const int16_t* input,
                int input_scale_shift);

void FftInit(struct FftState* state);

void FftReset(struct FftState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FFT_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_H_

#include <stdint.h>
#include <stdlib.h>

// #include "microfrontend/lib/fft.h"

#define kFilterbankBits 12

#ifdef __cplusplus
extern "C" {
#endif

struct FilterbankState {
  int num_channels;
  int start_index;
  int end_index;
  int16_t* channel_frequency_starts;
  int16_t* channel_weight_starts;
  int16_t* channel_widths;
  int16_t* weights;
  int16_t* unweights;
  uint64_t* work;
};

// Converts the relevant complex values of an FFT output into energy (the
// square magnitude).
void FilterbankConvertFftComplexToEnergy(struct FilterbankState* state,
                                         struct complex_int16_t* fft_output,
                                         int32_t* energy);

// Computes the mel-scale filterbank on the given energy array. Output is cached
// internally - to fetch it, you need to call FilterbankSqrt.
void FilterbankAccumulateChannels(struct FilterbankState* state,
                                  const int32_t* energy);

// Applies an integer square root to the 64 bit intermediate values of the
// filterbank, and returns a pointer to them. Memory will be invalidated the
// next time FilterbankAccumulateChannels is called.
uint32_t* FilterbankSqrt(struct FilterbankState* state, int scale_down_shift);

void FilterbankReset(struct FilterbankState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_H_

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LogScaleState {
  int enable_log;
  int scale_shift;
};

// Applies a fixed point logarithm to the signal and converts it to 16 bit. Note
// that the signal array will be modified.
uint16_t* LogScaleApply(struct LogScaleState* state, uint32_t* signal,
                        int signal_size, int correction_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_H_

#define kNoiseReductionBits 14

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct NoiseReductionState {
  int smoothing_bits;
  uint16_t even_smoothing;
  uint16_t odd_smoothing;
  uint16_t min_signal_remaining;
  int num_channels;
  uint32_t* estimate;
};

// Removes stationary noise from each channel of the signal using a low pass
// filter.
void NoiseReductionApply(struct NoiseReductionState* state, uint32_t* signal);

void NoiseReductionReset(struct NoiseReductionState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_H_

#include <stdint.h>
#include <stdlib.h>

#define kPcanSnrBits 12
#define kPcanOutputBits 6

#ifdef __cplusplus
extern "C" {
#endif

// Details at https://research.google/pubs/pub45911.pdf
struct PcanGainControlState {
  int enable_pcan;
  uint32_t* noise_estimate;
  int num_channels;
  int16_t* gain_lut;
  int32_t snr_shift;
};

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut);

uint32_t PcanShrink(const uint32_t x);

void PcanGainControlApply(struct PcanGainControlState* state, uint32_t* signal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_H_

#include <stdint.h>
#include <stdlib.h>

#define kFrontendWindowBits 12

#ifdef __cplusplus
extern "C" {
#endif

struct WindowState {
  size_t size;
  int16_t* coefficients;
  size_t step;

  int16_t* input;
  size_t input_used;
  int16_t* output;
  int16_t max_abs_output_value;
};

// Applies a window to the samples coming in, stepping forward at the given
// rate.
int WindowProcessSamples(struct WindowState* state, const int16_t* samples,
                         size_t num_samples, size_t* num_samples_read);

void WindowReset(struct WindowState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_H_

// #include "microfrontend/lib/frontend.h"
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_H_

#include <stdint.h>
#include <stdlib.h>

// #include "microfrontend/lib/fft.h"
// #include "microfrontend/lib/filterbank.h"
// #include "microfrontend/lib/log_scale.h"
// #include "microfrontend/lib/noise_reduction.h"
// #include "microfrontend/lib/pcan_gain_control.h"
// #include "microfrontend/lib/window.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FrontendState {
  struct WindowState window;
  struct FftState fft;
  struct FilterbankState filterbank;
  struct NoiseReductionState noise_reduction;
  struct PcanGainControlState pcan_gain_control;
  struct LogScaleState log_scale;
};

struct FrontendOutput {
  const uint16_t* values;
  size_t size;
};

// Main entry point to processing frontend samples. Updates num_samples_read to
// contain the number of samples that have been consumed from the input array.
// Returns a struct containing the generated output. If not enough samples were
// added to generate a feature vector, the returned size will be 0 and the
// values pointer will be NULL. Note that the output pointer will be invalidated
// as soon as FrontendProcessSamples is called again, so copy the contents
// elsewhere if you need to use them later.
struct FrontendOutput FrontendProcessSamples(struct FrontendState* state,
                                             const int16_t* samples,
                                             size_t num_samples,
                                             size_t* num_samples_read);

void FrontendReset(struct FrontendState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_BITS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_BITS_H_

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#endif

static inline int CountLeadingZeros32Slow(uint64_t n) {
  int zeroes = 28;
  if (n >> 16) zeroes -= 16, n >>= 16;
  if (n >> 8) zeroes -= 8, n >>= 8;
  if (n >> 4) zeroes -= 4, n >>= 4;
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[n] + zeroes;
}

static inline int CountLeadingZeros32(uint32_t n) {
#if defined(_MSC_VER)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse(&result, n)) {
    return 31 - result;
  }
  return 32;
#elif defined(__GNUC__)

  // Handle 0 as a special case because __builtin_clz(0) is undefined.
  if (n == 0) {
    return 32;
  }
  return __builtin_clz(n);
#else
  return CountLeadingZeros32Slow(n);
#endif
}

static inline int MostSignificantBit32(uint32_t n) {
  return 32 - CountLeadingZeros32(n);
}

static inline int CountLeadingZeros64Slow(uint64_t n) {
  int zeroes = 60;
  if (n >> 32) zeroes -= 32, n >>= 32;
  if (n >> 16) zeroes -= 16, n >>= 16;
  if (n >> 8) zeroes -= 8, n >>= 8;
  if (n >> 4) zeroes -= 4, n >>= 4;
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[n] + zeroes;
}

static inline int CountLeadingZeros64(uint64_t n) {
#if defined(_MSC_VER) && defined(_M_X64)
  // MSVC does not have __builtin_clzll. Use _BitScanReverse64.
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse64(&result, n)) {
    return 63 - result;
  }
  return 64;
#elif defined(_MSC_VER)
  // MSVC does not have __builtin_clzll. Compose two calls to _BitScanReverse
  unsigned long result = 0;  // NOLINT(runtime/int)
  if ((n >> 32) && _BitScanReverse(&result, n >> 32)) {
    return 31 - result;
  }
  if (_BitScanReverse(&result, n)) {
    return 63 - result;
  }
  return 64;
#elif defined(__GNUC__)

  // Handle 0 as a special case because __builtin_clzll(0) is undefined.
  if (n == 0) {
    return 64;
  }
  return __builtin_clzll(n);
#else
  return CountLeadingZeros64Slow(n);
#endif
}

static inline int MostSignificantBit64(uint64_t n) {
  return 64 - CountLeadingZeros64(n);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_BITS_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Number of segments in the log lookup table. The table will be kLogSegments+1
// in length (with some padding).
#define kLogSegments 128
#define kLogSegmentsLog2 7

// Scale used by lookup table.
#define kLogScale 65536
#define kLogScaleLog2 16
#define kLogCoeff 45426

extern const uint16_t kLogLut[];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_


#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_COMMON_H_

// This header file should be included in all variants of kiss_fft_$type.{h,cc}
// so that their sub-included source files do not mistakenly wrap libc header
// files within their kissfft_$type namespaces.
// E.g, This header avoids kissfft_int16.h containing:
//   namespace kiss_fft_int16 {
//     #include "kiss_fft.h"
//   }
// where kiss_fft_.h contains:
//   #include <math.h>
//
// TRICK: By including the following header files here, their preprocessor
// header guards prevent them being re-defined inside of the kiss_fft_$type
// namespaces declared within the kiss_fft_$type.{h,cc} sources.
// Note that the original kiss_fft*.h files are untouched since they
// may be used in libraries that include them directly.

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef FIXED_POINT
#include <sys/types.h>
#endif

#ifdef USE_SIMD
#include <xmmintrin.h>
#endif
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_KISS_FFT_COMMON_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_UTIL_H_

// #include "microfrontend/lib/filterbank.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FilterbankConfig {
  // number of frequency channel buckets for filterbank
  int num_channels;
  // maximum frequency to include
  float upper_band_limit;
  // minimum frequency to include
  float lower_band_limit;
  // unused
  int output_scale_shift;
};

// Fills the frontendConfig with "sane" defaults.
void FilterbankFillConfigWithDefaults(struct FilterbankConfig* config);

// Allocates any buffers.
int FilterbankPopulateState(const struct FilterbankConfig* config,
                            struct FilterbankState* state, int sample_rate,
                            int spectrum_size);

// Frees any allocated buffers.
void FilterbankFreeStateContents(struct FilterbankState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_UTIL_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_UTIL_H_

// #include "microfrontend/lib/window.h"

#ifdef __cplusplus
extern "C" {
#endif

struct WindowConfig {
  // length of window frame in milliseconds
  size_t size_ms;
  // length of step for next frame in milliseconds
  size_t step_size_ms;
};

// Populates the WindowConfig with "sane" default values.
void WindowFillConfigWithDefaults(struct WindowConfig* config);

// Allocates any buffers.
int WindowPopulateState(const struct WindowConfig* config,
                        struct WindowState* state, int sample_rate);

// Frees any allocated buffers.
void WindowFreeStateContents(struct WindowState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_UTIL_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_UTIL_H_

#include <stdint.h>
#include <stdlib.h>

// #include "microfrontend/lib/log_scale.h"

#ifdef __cplusplus
extern "C" {
#endif

struct LogScaleConfig {
  // set to false (0) to disable this module
  int enable_log;
  // scale results by 2^(scale_shift)
  int scale_shift;
};

// Populates the LogScaleConfig with "sane" default values.
void LogScaleFillConfigWithDefaults(struct LogScaleConfig* config);

// Allocates any buffers.
int LogScalePopulateState(const struct LogScaleConfig* config,
                          struct LogScaleState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_UTIL_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_UTIL_H_

// #include "microfrontend/lib/pcan_gain_control.h"

#define kWideDynamicFunctionBits 32
#define kWideDynamicFunctionLUTSize (4 * kWideDynamicFunctionBits - 3)

#ifdef __cplusplus
extern "C" {
#endif

struct PcanGainControlConfig {
  // set to false (0) to disable this module
  int enable_pcan;
  // gain normalization exponent (0.0 disables, 1.0 full strength)
  float strength;
  // positive value added in the normalization denominator
  float offset;
  // number of fractional bits in the gain
  int gain_bits;
};

void PcanGainControlFillConfigWithDefaults(
    struct PcanGainControlConfig* config);

int16_t PcanGainLookupFunction(const struct PcanGainControlConfig* config,
                               int32_t input_bits, uint32_t x);

int PcanGainControlPopulateState(const struct PcanGainControlConfig* config,
                                 struct PcanGainControlState* state,
                                 uint32_t* noise_estimate,
                                 const int num_channels,
                                 const uint16_t smoothing_bits,
                                 const int32_t input_correction_bits);

void PcanGainControlFreeStateContents(struct PcanGainControlState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_PCAN_GAIN_CONTROL_UTIL_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_

// #include "microfrontend/lib/noise_reduction.h"

#ifdef __cplusplus
extern "C" {
#endif

struct NoiseReductionConfig {
  // scale the signal up by 2^(smoothing_bits) before reduction
  int smoothing_bits;
  // smoothing coefficient for even-numbered channels
  float even_smoothing;
  // smoothing coefficient for odd-numbered channels
  float odd_smoothing;
  // fraction of signal to preserve (1.0 disables this module)
  float min_signal_remaining;
};

// Populates the NoiseReductionConfig with "sane" default values.
void NoiseReductionFillConfigWithDefaults(struct NoiseReductionConfig* config);

// Allocates any buffers.
int NoiseReductionPopulateState(const struct NoiseReductionConfig* config,
                                struct NoiseReductionState* state,
                                int num_channels);

// Frees any allocated buffers.
void NoiseReductionFreeStateContents(struct NoiseReductionState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_NOISE_REDUCTION_UTIL_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_UTIL_H_

/*#include "microfrontend/lib/fft_util.h"
#include "microfrontend/lib/filterbank_util.h"
#include "microfrontend/lib/frontend.h"
#include "microfrontend/lib/log_scale_util.h"
#include "microfrontend/lib/noise_reduction_util.h"
#include "microfrontend/lib/pcan_gain_control_util.h"
#include "microfrontend/lib/window_util.h"*/

#ifdef __cplusplus
extern "C" {
#endif

struct FrontendConfig {
  struct WindowConfig window;
  struct FilterbankConfig filterbank;
  struct NoiseReductionConfig noise_reduction;
  struct PcanGainControlConfig pcan_gain_control;
  struct LogScaleConfig log_scale;
};

// Fills the frontendConfig with "sane" defaults.
void FrontendFillConfigWithDefaults(struct FrontendConfig* config);

// Allocates any buffers.
int FrontendPopulateState(const struct FrontendConfig* config,
                          struct FrontendState* state, int sample_rate);

// Frees any allocated buffers.
void FrontendFreeStateContents(struct FrontendState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FRONTEND_UTIL_H_

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_H_

#include <stdint.h>
#include <stdlib.h>

// #include "microfrontend/lib/fft.h"

#define kFilterbankBits 12

#ifdef __cplusplus
extern "C" {
#endif

struct FilterbankState {
  int num_channels;
  int start_index;
  int end_index;
  int16_t* channel_frequency_starts;
  int16_t* channel_weight_starts;
  int16_t* channel_widths;
  int16_t* weights;
  int16_t* unweights;
  uint64_t* work;
};

// Converts the relevant complex values of an FFT output into energy (the
// square magnitude).
void FilterbankConvertFftComplexToEnergy(struct FilterbankState* state,
                                         struct complex_int16_t* fft_output,
                                         int32_t* energy);

// Computes the mel-scale filterbank on the given energy array. Output is cached
// internally - to fetch it, you need to call FilterbankSqrt.
void FilterbankAccumulateChannels(struct FilterbankState* state,
                                  const int32_t* energy);

// Applies an integer square root to the 64 bit intermediate values of the
// filterbank, and returns a pointer to them. Memory will be invalidated the
// next time FilterbankAccumulateChannels is called.
uint32_t* FilterbankSqrt(struct FilterbankState* state, int scale_down_shift);

void FilterbankReset(struct FilterbankState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_H_

// #include "microfrontend/lib/filterbank_util.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#define kFilterbankIndexAlignment 4
#define kFilterbankChannelBlockSize 4

void FilterbankFillConfigWithDefaults(struct FilterbankConfig* config) {
  config->num_channels = 32;
  config->lower_band_limit = 125.0f;
  config->upper_band_limit = 7500.0f;
  config->output_scale_shift = 7;
}

static float FreqToMel(float freq) { return 1127.0 * log1p(freq / 700.0); }

static void CalculateCenterFrequencies(const int num_channels,
                                       const float lower_frequency_limit,
                                       const float upper_frequency_limit,
                                       float* center_frequencies) {
  assert(lower_frequency_limit >= 0.0f);
  assert(upper_frequency_limit > lower_frequency_limit);

  const float mel_low = FreqToMel(lower_frequency_limit);
  const float mel_hi = FreqToMel(upper_frequency_limit);
  const float mel_span = mel_hi - mel_low;
  const float mel_spacing = mel_span / ((float)num_channels);
  int i;
  for (i = 0; i < num_channels; ++i) {
    center_frequencies[i] = mel_low + (mel_spacing * (i + 1));
  }
}

static void QuantizeFilterbankWeights(const float float_weight, int16_t* weight,
                                      int16_t* unweight) {
  *weight = floor(float_weight * (1 << kFilterbankBits) + 0.5);
  *unweight = floor((1.0 - float_weight) * (1 << kFilterbankBits) + 0.5);
}

#define MAX_CHANNELS 128  // choose a safe upper bound

static int16_t channel_frequency_starts_buf[MAX_CHANNELS];
static int16_t channel_weight_starts_buf[MAX_CHANNELS];
static int16_t channel_widths_buf[MAX_CHANNELS];
static uint64_t work_buf[MAX_CHANNELS];

static float center_mel_freqs[MAX_CHANNELS];
static int16_t actual_channel_starts[MAX_CHANNELS];
static int16_t actual_channel_widths[MAX_CHANNELS];

int FilterbankPopulateState(const struct FilterbankConfig* config,
                            struct FilterbankState* state, int sample_rate,
                            int spectrum_size) {
  state->num_channels = config->num_channels;
  const int num_channels_plus_1 = config->num_channels + 1;

  // How should we align things to index counts given the byte alignment?
  const int index_alignment =
      (kFilterbankIndexAlignment < sizeof(int16_t)
           ? 1
           : kFilterbankIndexAlignment / sizeof(int16_t));

  // state->channel_frequency_starts = (int16_t *)
  //     malloc(num_channels_plus_1 * sizeof(*state->channel_frequency_starts));
  // state->channel_weight_starts = (int16_t *)
  //     malloc(num_channels_plus_1 * sizeof(*state->channel_weight_starts));
  // state->channel_widths = (int16_t *)
  //     malloc(num_channels_plus_1 * sizeof(*state->channel_widths));
  // state->work = (uint64_t *) malloc(num_channels_plus_1 * sizeof(*state->work));

  state->channel_frequency_starts = channel_frequency_starts_buf;
  state->channel_weight_starts    = channel_weight_starts_buf;
  state->channel_widths           = channel_widths_buf;
  state->work                     = work_buf;

  float* center_mel_freqs = (float *)
      malloc(num_channels_plus_1 * sizeof(*center_mel_freqs));
  int16_t* actual_channel_starts = (int16_t*)
      malloc(num_channels_plus_1 * sizeof(*actual_channel_starts));
  int16_t* actual_channel_widths = (int16_t*)
      malloc(num_channels_plus_1 * sizeof(*actual_channel_widths));

  if (state->channel_frequency_starts == NULL ||
      state->channel_weight_starts == NULL || state->channel_widths == NULL ||
      center_mel_freqs == NULL || actual_channel_starts == NULL ||
      actual_channel_widths == NULL) {
    free(center_mel_freqs);
    free(actual_channel_starts);
    free(actual_channel_widths);
    printf("Failed to allocate channel buffers\n");
    return 0;
  }

  CalculateCenterFrequencies(num_channels_plus_1, config->lower_band_limit,
                             config->upper_band_limit, center_mel_freqs);

  // Always exclude DC.
  const float hz_per_sbin = 0.5 * sample_rate / ((float)spectrum_size - 1);
  state->start_index = 1.5 + config->lower_band_limit / hz_per_sbin;
  state->end_index = 0;  // Initialized to zero here, but actually set below.

  // For each channel, we need to figure out what frequencies belong to it, and
  // how much padding we need to add so that we can efficiently multiply the
  // weights and unweights for accumulation. To simplify the multiplication
  // logic, all channels will have some multiplication to do (even if there are
  // no frequencies that accumulate to that channel) - they will be directed to
  // a set of zero weights.
  int chan_freq_index_start = state->start_index;
  int weight_index_start = 0;
  int needs_zeros = 0;

  int chan;
  for (chan = 0; chan < num_channels_plus_1; ++chan) {
    // Keep jumping frequencies until we overshoot the bound on this channel.
    int freq_index = chan_freq_index_start;
    while (FreqToMel((freq_index)*hz_per_sbin) <= center_mel_freqs[chan]) {
      ++freq_index;
    }

    const int width = freq_index - chan_freq_index_start;
    actual_channel_starts[chan] = chan_freq_index_start;
    actual_channel_widths[chan] = width;

    if (width == 0) {
      // This channel doesn't actually get anything from the frequencies, it's
      // always zero. We need then to insert some 'zero' weights into the
      // output, and just redirect this channel to do a single multiplication at
      // this point. For simplicity, the zeros are placed at the beginning of
      // the weights arrays, so we have to go and update all the other
      // weight_starts to reflect this shift (but only once).
      state->channel_frequency_starts[chan] = 0;
      state->channel_weight_starts[chan] = 0;
      state->channel_widths[chan] = kFilterbankChannelBlockSize;
      if (!needs_zeros) {
        needs_zeros = 1;
        int j;
        for (j = 0; j < chan; ++j) {
          state->channel_weight_starts[j] += kFilterbankChannelBlockSize;
        }
        weight_index_start += kFilterbankChannelBlockSize;
      }
    } else {
      // How far back do we need to go to ensure that we have the proper
      // alignment?
      const int aligned_start =
          (chan_freq_index_start / index_alignment) * index_alignment;
      const int aligned_width = (chan_freq_index_start - aligned_start + width);
      const int padded_width =
          (((aligned_width - 1) / kFilterbankChannelBlockSize) + 1) *
          kFilterbankChannelBlockSize;

      state->channel_frequency_starts[chan] = aligned_start;
      state->channel_weight_starts[chan] = weight_index_start;
      state->channel_widths[chan] = padded_width;
      weight_index_start += padded_width;
    }
    chan_freq_index_start = freq_index;
  }

  // Allocate the two arrays to store the weights - weight_index_start contains
  // the index of what would be the next set of weights that we would need to
  // add, so that's how many weights we need to allocate.
  state->weights = (int16_t *) calloc(weight_index_start, sizeof(*state->weights));
  state->unweights = (int16_t *) calloc(weight_index_start, sizeof(*state->unweights));

  // If the alloc failed, we also need to nuke the arrays.
  if (state->weights == NULL || state->unweights == NULL) {
    free(center_mel_freqs);
    free(actual_channel_starts);
    free(actual_channel_widths);
    printf("Failed to allocate weights or unweights\n");
    return 0;
  }

  // Next pass, compute all the weights. Since everything has been memset to
  // zero, we only need to fill in the weights that correspond to some frequency
  // for a channel.
  const float mel_low = FreqToMel(config->lower_band_limit);
  for (chan = 0; chan < num_channels_plus_1; ++chan) {
    int frequency = actual_channel_starts[chan];
    const int num_frequencies = actual_channel_widths[chan];
    const int frequency_offset =
        frequency - state->channel_frequency_starts[chan];
    const int weight_start = state->channel_weight_starts[chan];
    const float denom_val = (chan == 0) ? mel_low : center_mel_freqs[chan - 1];

    int j;
    for (j = 0; j < num_frequencies; ++j, ++frequency) {
      const float weight =
          (center_mel_freqs[chan] - FreqToMel(frequency * hz_per_sbin)) /
          (center_mel_freqs[chan] - denom_val);

      // Make the float into an integer for the weights (and unweights).
      const int weight_index = weight_start + frequency_offset + j;
      QuantizeFilterbankWeights(weight, state->weights + weight_index,
                                state->unweights + weight_index);
    }
    if (frequency > state->end_index) {
      state->end_index = frequency;
    }
  }

  free(center_mel_freqs);
  free(actual_channel_starts);
  free(actual_channel_widths);
  if (state->end_index >= spectrum_size) {
    printf("Filterbank end_index is above spectrum size.\n");
    return 0;
  }
  return 1;
}

void FilterbankFreeStateContents(struct FilterbankState* state) {
  free(state->channel_frequency_starts);
  free(state->channel_weight_starts);
  free(state->channel_widths);
  free(state->weights);
  free(state->unweights);
  free(state->work);
}

// #include "microfrontend/lib/fft_util.h"
// #include "microfrontend/lib/kiss_fft_int16.h"

#define FIXED_POINT 16
namespace kissfft_fixed16 {
#ifndef _KISS_FFT_GUTS_H
#define _KISS_FFT_GUTS_H

/*
Copyright (c) 2003-2010, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* kiss_fft.h
   defines kiss_fft_scalar as either short or a float type
   and defines
   typedef struct { kiss_fft_scalar r; kiss_fft_scalar i; }kiss_fft_cpx; */
#ifndef KISS_FFT_H
#define KISS_FFT_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C++" {
#endif

/*
 ATTENTION!
 If you would like a :
 -- a utility that will handle the caching of fft objects
 -- real-only (no imaginary time component ) FFT
 -- a multi-dimensional FFT
 -- a command-line utility to perform ffts
 -- a command-line utility to perform fast-convolution filtering

 Then see kfc.h kiss_fftr.h kiss_fftnd.h fftutil.c kiss_fastfir.c
  in the tools/ directory.
*/

// #define USE_SIMD
#ifdef USE_SIMD
# include <xmmintrin.h>
# define kiss_fft_scalar __m128
#define KISS_FFT_MALLOC(nbytes) _mm_malloc(nbytes,16)
#define KISS_FFT_FREE _mm_free
#else
#define KISS_FFT_MALLOC(X) (void*)(0x0) /* Patched. */
#define KISS_FFT_FREE(X) /* Patched. */
#endif


#ifdef FIXED_POINT
#include <stdint.h> /* Patched. */
# if (FIXED_POINT == 32)
#  define kiss_fft_scalar int32_t
# else
#  define kiss_fft_scalar int16_t
# endif
#else
# ifndef kiss_fft_scalar
/*  default is float */
#   define kiss_fft_scalar float
# endif
#endif

typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
}kiss_fft_cpx;

typedef struct kiss_fft_state* kiss_fft_cfg;

/*
 *  kiss_fft_alloc
 *
 *  Initialize a FFT (or IFFT) algorithm's cfg/state buffer.
 *
 *  typical usage:      kiss_fft_cfg mycfg=kiss_fft_alloc(1024,0,NULL,NULL);
 *
 *  The return value from fft_alloc is a cfg buffer used internally
 *  by the fft routine or NULL.
 *
 *  If lenmem is NULL, then kiss_fft_alloc will allocate a cfg buffer using malloc.
 *  The returned value should be free()d when done to avoid memory leaks.
 *
 *  The state can be placed in a user supplied buffer 'mem':
 *  If lenmem is not NULL and mem is not NULL and *lenmem is large enough,
 *      then the function places the cfg in mem and the size used in *lenmem
 *      and returns mem.
 *
 *  If lenmem is not NULL and ( mem is NULL or *lenmem is not large enough),
 *      then the function returns NULL and places the minimum cfg
 *      buffer size in *lenmem.
 * */

kiss_fft_cfg kiss_fft_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem);

/*
 * kiss_fft(cfg,in_out_buf)
 *
 * Perform an FFT on a complex input buffer.
 * for a forward FFT,
 * fin should be  f[0] , f[1] , ... ,f[nfft-1]
 * fout will be   F[0] , F[1] , ... ,F[nfft-1]
 * Note that each element is complex and can be accessed like
    f[k].r and f[k].i
 * */
void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout);

/*
 A more generic version of the above function. It reads its input from every Nth sample.
 * */
void kiss_fft_stride(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int fin_stride);

/* If kiss_fft_alloc allocated a buffer, it is one contiguous
   buffer and can be simply free()d when no longer needed*/
#define kiss_fft_free free

/*
 Cleans up some memory that gets managed internally. Not necessary to call, but it might clean up
 your compiler output to call this before you exit.
*/
void kiss_fft_cleanup(void);


/*
 * Returns the smallest integer k, such that k>=n and k has only "fast" factors (2,3,5)
 */
int kiss_fft_next_fast_size(int n);

/* for real ffts, we need an even size */
#define kiss_fftr_next_fast_size_real(n) \
        (kiss_fft_next_fast_size( ((n)+1)>>1)<<1)

#ifdef __cplusplus
}
#endif

#endif
#include <limits.h>

#define MAXFACTORS 32
/* e.g. an fft of length 128 has 4 factors
 as far as kissfft is concerned
 4*4*4*2
 */

struct kiss_fft_state{
    int nfft;
    int inverse;
    int factors[2*MAXFACTORS];
    kiss_fft_cpx twiddles[1];
};

/*
  Explanation of macros dealing with complex math:

   C_MUL(m,a,b)         : m = a*b
   C_FIXDIV( c , div )  : if a fixed point impl., c /= div. noop otherwise
   C_SUB( res, a,b)     : res = a - b
   C_SUBFROM( res , a)  : res -= a
   C_ADDTO( res , a)    : res += a
 * */
#ifdef FIXED_POINT
#if (FIXED_POINT==32)
# define FRACBITS 31
# define SAMPPROD int64_t
#define SAMP_MAX 2147483647
#else
# define FRACBITS 15
# define SAMPPROD int32_t
#define SAMP_MAX 32767
#endif

#define SAMP_MIN -SAMP_MAX

#if defined(CHECK_OVERFLOW)
#  define CHECK_OVERFLOW_OP(a,op,b)  \
	if ( (SAMPPROD)(a) op (SAMPPROD)(b) > SAMP_MAX || (SAMPPROD)(a) op (SAMPPROD)(b) < SAMP_MIN ) { \
		printf(@ " __FILE__ "(%d): (%d " #op" %d) = %ld\n",__LINE__,(a),(b),(SAMPPROD)(a) op (SAMPPROD)(b) );  }
#endif


#   define smul(a,b) ( (SAMPPROD)(a)*(b) )
#   define sround( x )  (kiss_fft_scalar)( ( (x) + (1<<(FRACBITS-1)) ) >> FRACBITS )

#   define S_MUL(a,b) sround( smul(a,b) )

#   define C_MUL(m,a,b) \
      do{ (m).r = sround( smul((a).r,(b).r) - smul((a).i,(b).i) ); \
          (m).i = sround( smul((a).r,(b).i) + smul((a).i,(b).r) ); }while(0)

#   define DIVSCALAR(x,k) \
	(x) = sround( smul(  x, SAMP_MAX/k ) )

#   define C_FIXDIV(c,div) \
	do {    DIVSCALAR( (c).r , div);  \
		DIVSCALAR( (c).i  , div); }while (0)

#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r =  sround( smul( (c).r , s ) ) ;\
        (c).i =  sround( smul( (c).i , s ) ) ; }while(0)

#else  /* not FIXED_POINT*/

#   define S_MUL(a,b) ( (a)*(b) )
#define C_MUL(m,a,b) \
    do{ (m).r = (a).r*(b).r - (a).i*(b).i;\
        (m).i = (a).r*(b).i + (a).i*(b).r; }while(0)
#   define C_FIXDIV(c,div) /* NOOP */
#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r *= (s);\
        (c).i *= (s); }while(0)
#endif

#ifndef CHECK_OVERFLOW_OP
#  define CHECK_OVERFLOW_OP(a,op,b) /* noop */
#endif

#define  C_ADD( res, a,b)\
    do { \
	    CHECK_OVERFLOW_OP((a).r,+,(b).r)\
	    CHECK_OVERFLOW_OP((a).i,+,(b).i)\
	    (res).r=(a).r+(b).r;  (res).i=(a).i+(b).i; \
    }while(0)
#define  C_SUB( res, a,b)\
    do { \
	    CHECK_OVERFLOW_OP((a).r,-,(b).r)\
	    CHECK_OVERFLOW_OP((a).i,-,(b).i)\
	    (res).r=(a).r-(b).r;  (res).i=(a).i-(b).i; \
    }while(0)
#define C_ADDTO( res , a)\
    do { \
	    CHECK_OVERFLOW_OP((res).r,+,(a).r)\
	    CHECK_OVERFLOW_OP((res).i,+,(a).i)\
	    (res).r += (a).r;  (res).i += (a).i;\
    }while(0)

#define C_SUBFROM( res , a)\
    do {\
	    CHECK_OVERFLOW_OP((res).r,-,(a).r)\
	    CHECK_OVERFLOW_OP((res).i,-,(a).i)\
	    (res).r -= (a).r;  (res).i -= (a).i; \
    }while(0)


#ifdef FIXED_POINT
#  define KISS_FFT_COS(phase)  floor(.5+SAMP_MAX * cos (phase))
#  define KISS_FFT_SIN(phase)  floor(.5+SAMP_MAX * sin (phase))
#  define HALF_OF(x) ((x)>>1)
#elif defined(USE_SIMD)
#  define KISS_FFT_COS(phase) _mm_set1_ps( cos(phase) )
#  define KISS_FFT_SIN(phase) _mm_set1_ps( sin(phase) )
#  define HALF_OF(x) ((x)*_mm_set1_ps(.5))
#else
#  define KISS_FFT_COS(phase) (kiss_fft_scalar) cos(phase)
#  define KISS_FFT_SIN(phase) (kiss_fft_scalar) sin(phase)
#  define HALF_OF(x) ((x)*(kiss_fft_scalar).5)
#endif

#define  kf_cexp(x,phase) \
	do{ \
		(x)->r = KISS_FFT_COS(phase);\
		(x)->i = KISS_FFT_SIN(phase);\
	}while(0)


/* a debugging function */
// #define pcpx(c)\
//     printf(+ %gi\n",(double)((c)->r),(double)((c)->i) )


#ifdef KISS_FFT_USE_ALLOCA
// define this to allow use of alloca instead of malloc for temporary buffers
// Temporary buffers are used in two case:
// 1. FFT sizes that have "bad" factors. i.e. not 2,3 and 5
// 2. "in-place" FFTs.  Notice the quotes, since kissfft does not really do an in-place transform.
#include <alloca.h>
#define  KISS_FFT_TMP_ALLOC(nbytes) alloca(nbytes)
#define  KISS_FFT_TMP_FREE(ptr)
#else
#define  KISS_FFT_TMP_ALLOC(nbytes) KISS_FFT_MALLOC(nbytes)
#define  KISS_FFT_TMP_FREE(ptr) KISS_FFT_FREE(ptr)
#endif
#endif // _KISS_FFT_GUTS_H

/* The guts header contains all the multiplication and addition macros that are defined for
 fixed or floating point complex numbers.  It also delares the kf_ internal functions.
 */

static void kf_bfly2(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx * Fout2;
    kiss_fft_cpx * tw1 = st->twiddles;
    kiss_fft_cpx t;
    Fout2 = Fout + m;
    do{
        C_FIXDIV(*Fout,2); C_FIXDIV(*Fout2,2);

        C_MUL (t,  *Fout2 , *tw1);
        tw1 += fstride;
        C_SUB( *Fout2 ,  *Fout , t );
        C_ADDTO( *Fout ,  t );
        ++Fout2;
        ++Fout;
    }while (--m);
}

static void kf_bfly4(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        const size_t m
        )
{
    kiss_fft_cpx *tw1,*tw2,*tw3;
    kiss_fft_cpx scratch[6];
    size_t k=m;
    const size_t m2=2*m;
    const size_t m3=3*m;


    tw3 = tw2 = tw1 = st->twiddles;

    do {
        C_FIXDIV(*Fout,4); C_FIXDIV(Fout[m],4); C_FIXDIV(Fout[m2],4); C_FIXDIV(Fout[m3],4);

        C_MUL(scratch[0],Fout[m] , *tw1 );
        C_MUL(scratch[1],Fout[m2] , *tw2 );
        C_MUL(scratch[2],Fout[m3] , *tw3 );

        C_SUB( scratch[5] , *Fout, scratch[1] );
        C_ADDTO(*Fout, scratch[1]);
        C_ADD( scratch[3] , scratch[0] , scratch[2] );
        C_SUB( scratch[4] , scratch[0] , scratch[2] );
        C_SUB( Fout[m2], *Fout, scratch[3] );
        tw1 += fstride;
        tw2 += fstride*2;
        tw3 += fstride*3;
        C_ADDTO( *Fout , scratch[3] );

        if(st->inverse) {
            Fout[m].r = scratch[5].r - scratch[4].i;
            Fout[m].i = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        }else{
            Fout[m].r = scratch[5].r + scratch[4].i;
            Fout[m].i = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }
        ++Fout;
    }while(--k);
}

static void kf_bfly3(
         kiss_fft_cpx * Fout,
         const size_t fstride,
         const kiss_fft_cfg st,
         size_t m
         )
{
     size_t k=m;
     const size_t m2 = 2*m;
     kiss_fft_cpx *tw1,*tw2;
     kiss_fft_cpx scratch[5];
     kiss_fft_cpx epi3;
     epi3 = st->twiddles[fstride*m];

     tw1=tw2=st->twiddles;

     do{
         C_FIXDIV(*Fout,3); C_FIXDIV(Fout[m],3); C_FIXDIV(Fout[m2],3);

         C_MUL(scratch[1],Fout[m] , *tw1);
         C_MUL(scratch[2],Fout[m2] , *tw2);

         C_ADD(scratch[3],scratch[1],scratch[2]);
         C_SUB(scratch[0],scratch[1],scratch[2]);
         tw1 += fstride;
         tw2 += fstride*2;

         Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
         Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

         C_MULBYSCALAR( scratch[0] , epi3.i );

         C_ADDTO(*Fout,scratch[3]);

         Fout[m2].r = Fout[m].r + scratch[0].i;
         Fout[m2].i = Fout[m].i - scratch[0].r;

         Fout[m].r -= scratch[0].i;
         Fout[m].i += scratch[0].r;

         ++Fout;
     }while(--k);
}

static void kf_bfly5(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
    int u;
    kiss_fft_cpx scratch[13];
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx *tw;
    kiss_fft_cpx ya,yb;
    ya = twiddles[fstride*m];
    yb = twiddles[fstride*2*m];

    Fout0=Fout;
    Fout1=Fout0+m;
    Fout2=Fout0+2*m;
    Fout3=Fout0+3*m;
    Fout4=Fout0+4*m;

    tw=st->twiddles;
    for ( u=0; u<m; ++u ) {
        C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
        scratch[0] = *Fout0;

        C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
        C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
        C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
        C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

        C_ADD( scratch[7],scratch[1],scratch[4]);
        C_SUB( scratch[10],scratch[1],scratch[4]);
        C_ADD( scratch[8],scratch[2],scratch[3]);
        C_SUB( scratch[9],scratch[2],scratch[3]);

        Fout0->r += scratch[7].r + scratch[8].r;
        Fout0->i += scratch[7].i + scratch[8].i;

        scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
        scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

        scratch[6].r =  S_MUL(scratch[10].i,ya.i) + S_MUL(scratch[9].i,yb.i);
        scratch[6].i = -S_MUL(scratch[10].r,ya.i) - S_MUL(scratch[9].r,yb.i);

        C_SUB(*Fout1,scratch[5],scratch[6]);
        C_ADD(*Fout4,scratch[5],scratch[6]);

        scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
        scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
        scratch[12].r = - S_MUL(scratch[10].i,yb.i) + S_MUL(scratch[9].i,ya.i);
        scratch[12].i = S_MUL(scratch[10].r,yb.i) - S_MUL(scratch[9].r,ya.i);

        C_ADD(*Fout2,scratch[11],scratch[12]);
        C_SUB(*Fout3,scratch[11],scratch[12]);

        ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
    }
}

/* perform the butterfly for one stage of a mixed radix FFT */
static void kf_bfly_generic(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m,
        int p
        )
{
    int u,k,q1,q;
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx t;
    int Norig = st->nfft;

    kiss_fft_cpx * scratch = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(sizeof(kiss_fft_cpx)*p);

    for ( u=0; u<m; ++u ) {
        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            scratch[q1] = Fout[ k  ];
            C_FIXDIV(scratch[q1],p);
            k += m;
        }

        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            int twidx=0;
            Fout[ k ] = scratch[0];
            for (q=1;q<p;++q ) {
                twidx += fstride * k;
                if (twidx>=Norig) twidx-=Norig;
                C_MUL(t,scratch[q] , twiddles[twidx] );
                C_ADDTO( Fout[ k ] ,t);
            }
            k += m;
        }
    }
    KISS_FFT_TMP_FREE(scratch);
}

static
void kf_work(
        kiss_fft_cpx * Fout,
        const kiss_fft_cpx * f,
        const size_t fstride,
        int in_stride,
        int * factors,
        const kiss_fft_cfg st
        )
{
    kiss_fft_cpx * Fout_beg=Fout;
    const int p=*factors++; /* the radix  */
    const int m=*factors++; /* stage's fft length/p */
    const kiss_fft_cpx * Fout_end = Fout + p*m;

#ifdef _OPENMP
    // use openmp extensions at the
    // top-level (not recursive)
    if (fstride==1 && p<=5)
    {
        int k;

        // execute the p different work units in different threads
#       pragma omp parallel for
        for (k=0;k<p;++k)
            kf_work( Fout +k*m, f+ fstride*in_stride*k,fstride*p,in_stride,factors,st);
        // all threads have joined by this point

        switch (p) {
            case 2: kf_bfly2(Fout,fstride,st,m); break;
            case 3: kf_bfly3(Fout,fstride,st,m); break;
            case 4: kf_bfly4(Fout,fstride,st,m); break;
            case 5: kf_bfly5(Fout,fstride,st,m); break;
            default: kf_bfly_generic(Fout,fstride,st,m,p); break;
        }
        return;
    }
#endif

    if (m==1) {
        do{
            *Fout = *f;
            f += fstride*in_stride;
        }while(++Fout != Fout_end );
    }else{
        do{
            // recursive call:
            // DFT of size m*p performed by doing
            // p instances of smaller DFTs of size m,
            // each one takes a decimated version of the input
            kf_work( Fout , f, fstride*p, in_stride, factors,st);
            f += fstride*in_stride;
        }while( (Fout += m) != Fout_end );
    }

    Fout=Fout_beg;

    // recombine the p smaller DFTs
    switch (p) {
        case 2: kf_bfly2(Fout,fstride,st,m); break;
        case 3: kf_bfly3(Fout,fstride,st,m); break;
        case 4: kf_bfly4(Fout,fstride,st,m); break;
        case 5: kf_bfly5(Fout,fstride,st,m); break;
        default: kf_bfly_generic(Fout,fstride,st,m,p); break;
    }
}

/*  facbuf is populated by p1,m1,p2,m2, ...
    where
    p[i] * m[i] = m[i-1]
    m0 = n                  */
static
void kf_factor(int n,int * facbuf)
{
    int p=4;
    double floor_sqrt;
    floor_sqrt = floor( sqrt((double)n) );

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
}

/*
 *
 * User-callable function to allocate all necessary storage space for the fft.
 *
 * The return value is a contiguous block of memory, allocated with malloc.  As such,
 * It can be freed with free(), rather than a kiss_fft-specific function.
 * */
kiss_fft_cfg kiss_fft_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem )
{
    kiss_fft_cfg st=NULL;
    size_t memneeded = sizeof(struct kiss_fft_state)
        + sizeof(kiss_fft_cpx)*(nfft-1); /* twiddle factors*/

    if ( lenmem==NULL ) {
        st = ( kiss_fft_cfg)KISS_FFT_MALLOC( memneeded );
    }else{
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st) {
        int i;
        st->nfft=nfft;
        st->inverse = inverse_fft;

        for (i=0;i<nfft;++i) {
            const double pi=3.141592653589793238462643383279502884197169399375105820974944;
            double phase = -2*pi*i / nfft;
            if (st->inverse)
                phase *= -1;
            kf_cexp(st->twiddles+i, phase );
        }

        kf_factor(nfft,st->factors);
    }
    return st;
}


void kiss_fft_stride(kiss_fft_cfg st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int in_stride)
{
    if (fin == fout) {
        //NOTE: this is not really an in-place FFT algorithm.
        //It just performs an out-of-place FFT into a temp buffer
        kiss_fft_cpx * tmpbuf = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC( sizeof(kiss_fft_cpx)*st->nfft);
        kf_work(tmpbuf,fin,1,in_stride, st->factors,st);
        /* memcpy(fout,tmpbuf,sizeof(kiss_fft_cpx)*st->nfft); */
        KISS_FFT_TMP_FREE(tmpbuf);
    }else{
        kf_work( fout, fin, 1,in_stride, st->factors,st );
    }
}

void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
    kiss_fft_stride(cfg,fin,fout,1);
}


void kiss_fft_cleanup(void)
{
    // nothing needed any more
}

int kiss_fft_next_fast_size(int n)
{
    while(1) {
        int m=n;
        while ( (m%2) == 0 ) m/=2;
        while ( (m%3) == 0 ) m/=3;
        while ( (m%5) == 0 ) m/=5;
        if (m<=1)
            break; /* n is completely factorable by twos, threes, and fives */
        n++;
    }
    return n;
}
#ifndef KISS_FTR_H
#define KISS_FTR_H

#ifdef __cplusplus
extern "C" {
#endif


/*

 Real optimized version can save about 45% cpu time vs. complex fft of a real seq.



 */

typedef struct kiss_fftr_state *kiss_fftr_cfg;


kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void * mem, size_t * lenmem);
/*
 nfft must be even

 If you don't care to allocate space, use mem = lenmem = NULL
*/


void kiss_fftr(kiss_fftr_cfg cfg,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata);
/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points
*/

void kiss_fftri(kiss_fftr_cfg cfg,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata);
/*
 input freqdata has  nfft/2+1 complex points
 output timedata has nfft scalar points
*/

#define kiss_fftr_free free

#ifdef __cplusplus
}
#endif
#endif
// #include "kiss_fftr.h"
// #include "_kiss_fft_guts.h"

struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
#ifdef USE_SIMD
    void * pad;
#endif
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem)
{
    int i;
    kiss_fftr_cfg st = NULL;
    size_t subsize, memneeded;

    if (nfft & 1) {
        /* printf("Real FFT optimization must be even.\n"); */
        return NULL;
    }
    nfft >>= 1;

    kiss_fft_alloc (nfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state) + subsize + sizeof(kiss_fft_cpx) * ( nfft * 3 / 2);

    if (lenmem == NULL) {
        st = (kiss_fftr_cfg) KISS_FFT_MALLOC (memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_fftr_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->substate = (kiss_fft_cfg) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (kiss_fft_cpx *) (((char *) st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + nfft;
    kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

    for (i = 0; i < nfft/2; ++i) {
        double phase =
            -3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);
        if (inverse_fft)
            phase *= -1;
        kf_cexp (st->super_twiddles+i,phase);
    }
    return st;
}

void kiss_fftr(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    uint64_t t0 = rdcycle64();
    int k,ncfft;
    kiss_fft_cpx fpnk,fpk,f1k,f2k,tw,tdc;

    if ( st->substate->inverse) {
        /* printf("kiss fft usage error: improper alloc\n"); */
        return; /* exit(1); */
    }

    ncfft = st->substate->nfft;
    uint64_t t1 = rdcycle64();
    uint64_t t01 = t1 - t0;
    printf("t(fftr enter)=%llu\n", t01);

    /*perform the parallel fft of two real signals packed in real,imag*/
    uint64_t t0_ = rdcycle64();
    kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf );
    uint64_t t1_ = rdcycle64();
    uint64_t t01_ = t1_ - t0_;
    printf("t(kiss_fft)=%llu\n", t01_);
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    uint64_t t0__ = rdcycle64();
    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;
    C_FIXDIV(tdc,2);
    CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
    CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD
    freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#else
    freqdata[ncfft].i = freqdata[0].i = 0;
#endif

    for ( k=1;k <= ncfft/2 ; ++k ) {
        fpk    = st->tmpbuf[k];
        fpnk.r =   st->tmpbuf[ncfft-k].r;
        fpnk.i = - st->tmpbuf[ncfft-k].i;
        C_FIXDIV(fpk,2);
        C_FIXDIV(fpnk,2);

        C_ADD( f1k, fpk , fpnk );
        C_SUB( f2k, fpk , fpnk );
        C_MUL( tw , f2k , st->super_twiddles[k-1]);

        freqdata[k].r = HALF_OF(f1k.r + tw.r);
        freqdata[k].i = HALF_OF(f1k.i + tw.i);
        freqdata[ncfft-k].r = HALF_OF(f1k.r - tw.r);
        freqdata[ncfft-k].i = HALF_OF(tw.i - f1k.i);
    }
    uint64_t t1__ = rdcycle64();
    uint64_t t01__ = t1__ - t0__;
    printf("t(fftr rest)=%llu\n", t01__);
}

void kiss_fftri(kiss_fftr_cfg st,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata)
{
    /* input buffer timedata is stored row-wise */
    int k, ncfft;

    if (st->substate->inverse == 0) {
        /* printf("kiss fft usage error: improper alloc\n"); */
        return; /* exit (1); */
    }

    ncfft = st->substate->nfft;

    st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
    C_FIXDIV(st->tmpbuf[0],2);

    for (k = 1; k <= ncfft / 2; ++k) {
        kiss_fft_cpx fk, fnkc, fek, fok, tmp;
        fk = freqdata[k];
        fnkc.r = freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;
        C_FIXDIV( fk , 2 );
        C_FIXDIV( fnkc , 2 );

        C_ADD (fek, fk, fnkc);
        C_SUB (tmp, fk, fnkc);
        C_MUL (fok, tmp, st->super_twiddles[k-1]);
        C_ADD (st->tmpbuf[k],     fek, fok);
        C_SUB (st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD
        st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
        st->tmpbuf[ncfft - k].i *= -1;
#endif
    }
    kiss_fft (st->substate, st->tmpbuf, (kiss_fft_cpx *) timedata);
}
}  // namespace kissfft_fixed16
#undef FIXED_POINT

#include <stdio.h>

int FftPopulateState(struct FftState* state, size_t input_size) {
  state->input_size = input_size;
  state->fft_size = 1;
  while (state->fft_size < state->input_size) {
    state->fft_size <<= 1;
  }

  state->input = reinterpret_cast<int16_t*>(
      malloc(state->fft_size * sizeof(*state->input)));
  if (state->input == nullptr) {
    printf("Failed to alloc fft input buffer\n");
    return 0;
  }

  state->output = reinterpret_cast<complex_int16_t*>(
      malloc((state->fft_size / 2 + 1) * sizeof(*state->output) * 2));
  if (state->output == nullptr) {
    printf("Failed to alloc fft output buffer\n");
    return 0;
  }

  // Ask kissfft how much memory it wants.
  size_t scratch_size = 0;
  kissfft_fixed16::kiss_fftr_cfg kfft_cfg = kissfft_fixed16::kiss_fftr_alloc(
      state->fft_size, 0, nullptr, &scratch_size);
  if (kfft_cfg != nullptr) {
    printf("Kiss memory sizing failed.\n");
    return 0;
  }
  state->scratch = malloc(scratch_size);
  if (state->scratch == nullptr) {
    printf("Failed to alloc fft scratch buffer\n");
    return 0;
  }
  state->scratch_size = scratch_size;
  // Let kissfft configure the scratch space we just allocated
  kfft_cfg = kissfft_fixed16::kiss_fftr_alloc(state->fft_size, 0,
                                              state->scratch, &scratch_size);
  if (kfft_cfg != state->scratch) {
    printf("Kiss memory preallocation strategy failed.\n");
    return 0;
  }
  return 1;
}

void FftFreeStateContents(struct FftState* state) {
  free(state->input);
  free(state->output);
  free(state->scratch);
}

#include <string.h>

void NoiseReductionApply(struct NoiseReductionState* state, uint32_t* signal) {
  int i;
  for (i = 0; i < state->num_channels; ++i) {
    const uint32_t smoothing =
        ((i & 1) == 0) ? state->even_smoothing : state->odd_smoothing;
    const uint32_t one_minus_smoothing = (1 << kNoiseReductionBits) - smoothing;

    // Update the estimate of the noise.
    const uint32_t signal_scaled_up = signal[i] << state->smoothing_bits;
    uint32_t estimate =
        (((uint64_t)signal_scaled_up * smoothing) +
         ((uint64_t)state->estimate[i] * one_minus_smoothing)) >>
        kNoiseReductionBits;
    state->estimate[i] = estimate;

    // Make sure that we can't get a negative value for the signal - estimate.
    if (estimate > signal_scaled_up) {
      estimate = signal_scaled_up;
    }

    const uint32_t floor =
        ((uint64_t)signal[i] * state->min_signal_remaining) >>
        kNoiseReductionBits;
    const uint32_t subtracted =
        (signal_scaled_up - estimate) >> state->smoothing_bits;
    const uint32_t output = subtracted > floor ? subtracted : floor;
    signal[i] = output;
  }
}

void NoiseReductionReset(struct NoiseReductionState* state) {
  memset(state->estimate, 0, sizeof(*state->estimate) * state->num_channels);
}

int WindowProcessSamples(struct WindowState* state, const int16_t* samples,
                         size_t num_samples, size_t* num_samples_read) {
  const int size = state->size;

  // Copy samples from the samples buffer over to our local input.
  size_t max_samples_to_copy = state->size - state->input_used;
  if (max_samples_to_copy > num_samples) {
    max_samples_to_copy = num_samples;
  }
  memcpy(state->input + state->input_used, samples,
         max_samples_to_copy * sizeof(*samples));
  *num_samples_read = max_samples_to_copy;
  state->input_used += max_samples_to_copy;

  if (state->input_used < state->size) {
    // We don't have enough samples to compute a window.
    return 0;
  }

  // Apply the window to the input.
  const int16_t* coefficients = state->coefficients;
  const int16_t* input = state->input;
  int16_t* output = state->output;
  int i;
  int16_t max_abs_output_value = 0;
  for (i = 0; i < size; ++i) {
    int16_t new_value =
        (((int32_t)*input++) * *coefficients++) >> kFrontendWindowBits;
    *output++ = new_value;
    if (new_value < 0) {
      new_value = -new_value;
    }
    if (new_value > max_abs_output_value) {
      max_abs_output_value = new_value;
    }
  }
  // Shuffle the input down by the step size, and update how much we have used.
  memmove(state->input, state->input + state->step,
          sizeof(*state->input) * (state->size - state->step));
  state->input_used -= state->step;
  state->max_abs_output_value = max_abs_output_value;

  // Indicate that the output buffer is valid for the next stage.
  return 1;
}

void WindowReset(struct WindowState* state) {
  memset(state->input, 0, state->size * sizeof(*state->input));
  memset(state->output, 0, state->size * sizeof(*state->output));
  state->input_used = 0;
  state->max_abs_output_value = 0;
}

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut) {
  if (x <= 2) {
    return lut[x];
  }

  const int16_t interval = MostSignificantBit32(x);
  lut += 4 * interval - 6;

  const int16_t frac =
      ((interval < 11) ? (x << (11 - interval)) : (x >> (interval - 11))) &
      0x3FF;

  int32_t result = ((int32_t)lut[2] * frac) >> 5;
  result += (int32_t)((uint32_t)lut[1] << 5);
  result *= frac;
  result = (result + (1 << 14)) >> 15;
  result += lut[0];
  return (int16_t)result;
}

uint32_t PcanShrink(const uint32_t x) {
  if (x < (2 << kPcanSnrBits)) {
    return (x * x) >> (2 + 2 * kPcanSnrBits - kPcanOutputBits);
  } else {
    return (x >> (kPcanSnrBits - kPcanOutputBits)) - (1 << kPcanOutputBits);
  }
}

void PcanGainControlApply(struct PcanGainControlState* state,
                          uint32_t* signal) {
  int i;
  for (i = 0; i < state->num_channels; ++i) {
    const uint32_t gain =
        WideDynamicFunction(state->noise_estimate[i], state->gain_lut);
    const uint32_t snr = ((uint64_t)signal[i] * gain) >> state->snr_shift;
    signal[i] = PcanShrink(snr);
  }
}


// #include "microfrontend/lib/log_lut.h"
const uint16_t kLogLut[]
#ifndef _MSC_VER
    __attribute__((aligned(4)))
#endif  // _MSV_VER
    = {0,    224,  442,  654,  861,  1063, 1259, 1450, 1636, 1817, 1992, 2163,
       2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
       3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
       5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
       5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
       5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
       5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
       4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
       3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
       2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
       1094, 963,  830,  695,  559,  421,  282,  142,  0,    0};


#define kuint16max 0x0000FFFF

// The following functions implement integer logarithms of various sizes. The
// approximation is calculated according to method described in
//       www.inti.gob.ar/electronicaeinformatica/instrumentacion/utic/
//       publicaciones/SPL2007/Log10-spl07.pdf
// It first calculates log2 of the input and then converts it to natural
// logarithm.

static uint32_t Log2FractionPart(const uint32_t x, const uint32_t log2x) {
  // Part 1
  int32_t frac = x - (1LL << log2x);
  if (log2x < kLogScaleLog2) {
    frac <<= kLogScaleLog2 - log2x;
  } else {
    frac >>= log2x - kLogScaleLog2;
  }
  // Part 2
  const uint32_t base_seg = frac >> (kLogScaleLog2 - kLogSegmentsLog2);
  const uint32_t seg_unit =
      (((uint32_t)1) << kLogScaleLog2) >> kLogSegmentsLog2;

  const int32_t c0 = kLogLut[base_seg];
  const int32_t c1 = kLogLut[base_seg + 1];
  const int32_t seg_base = seg_unit * base_seg;
  const int32_t rel_pos = ((c1 - c0) * (frac - seg_base)) >> kLogScaleLog2;
  return frac + c0 + rel_pos;
}

static uint32_t Log(const uint32_t x, const uint32_t scale_shift) {
  const uint32_t integer = MostSignificantBit32(x) - 1;
  const uint32_t fraction = Log2FractionPart(x, integer);
  const uint32_t log2 = (integer << kLogScaleLog2) + fraction;
  const uint32_t round = kLogScale / 2;
  const uint32_t loge = (((uint64_t)kLogCoeff) * log2 + round) >> kLogScaleLog2;
  // Finally scale to our output scale
  const uint32_t loge_scaled = ((loge << scale_shift) + round) >> kLogScaleLog2;
  return loge_scaled;
}

uint16_t* LogScaleApply(struct LogScaleState* state, uint32_t* signal,
                        int signal_size, int correction_bits) {
  const int scale_shift = state->scale_shift;
  uint16_t* output = (uint16_t*)signal;
  uint16_t* ret = output;
  int i;
  for (i = 0; i < signal_size; ++i) {
    uint32_t value = *signal++;
    if (state->enable_log) {
      if (correction_bits < 0) {
        value >>= -correction_bits;
      } else {
        value <<= correction_bits;
      }
      if (value > 1) {
        value = Log(value, scale_shift);
      } else {
        value = 0;
      }
    }
    *output++ = (value < kuint16max) ? value : kuint16max;
  }
  return ret;
}

// #include "microfrontend/lib/filterbank.h"

#include <string.h>

// #include "microfrontend/lib/bits.h"

void FilterbankConvertFftComplexToEnergy(struct FilterbankState* state,
                                         struct complex_int16_t* fft_output,
                                         int32_t* energy) {
  const int end_index = state->end_index;
  int i;
  energy += state->start_index;
  fft_output += state->start_index;
  for (i = state->start_index; i < end_index; ++i) {
    const int32_t real = fft_output->real;
    const int32_t imag = fft_output->imag;
    fft_output++;
    const uint32_t mag_squared = (real * real) + (imag * imag);
    *energy++ = mag_squared;
  }
}

void FilterbankAccumulateChannels(struct FilterbankState* state,
                                  const int32_t* energy) {
  uint64_t* work = state->work;
  uint64_t weight_accumulator = 0;
  uint64_t unweight_accumulator = 0;

  const int16_t* channel_frequency_starts = state->channel_frequency_starts;
  const int16_t* channel_weight_starts = state->channel_weight_starts;
  const int16_t* channel_widths = state->channel_widths;

  int num_channels_plus_1 = state->num_channels + 1;
  int i;
  for (i = 0; i < num_channels_plus_1; ++i) {
    const int32_t* magnitudes = energy + *channel_frequency_starts++;
    const int16_t* weights = state->weights + *channel_weight_starts;
    const int16_t* unweights = state->unweights + *channel_weight_starts++;
    const int width = *channel_widths++;
    int j;
    for (j = 0; j < width; ++j) {
      weight_accumulator += *weights++ * ((uint64_t)*magnitudes);
      unweight_accumulator += *unweights++ * ((uint64_t)*magnitudes);
      ++magnitudes;
    }
    *work++ = weight_accumulator;
    weight_accumulator = unweight_accumulator;
    unweight_accumulator = 0;
  }
}

static uint16_t Sqrt32(uint32_t num) {
  if (num == 0) {
    return 0;
  }
  uint32_t res = 0;
  int max_bit_number = 32 - MostSignificantBit32(num);
  max_bit_number |= 1;
  uint32_t bit = 1U << (31 - max_bit_number);
  int iterations = (31 - max_bit_number) / 2 + 1;
  while (iterations--) {
    if (num >= res + bit) {
      num -= res + bit;
      res = (res >> 1U) + bit;
    } else {
      res >>= 1U;
    }
    bit >>= 2U;
  }
  // Do rounding - if we have the bits.
  if (num > res && res != 0xFFFF) {
    ++res;
  }
  return res;
}

static uint32_t Sqrt64(uint64_t num) {
  // Take a shortcut and just use 32 bit operations if the upper word is all
  // clear. This will cause a slight off by one issue for numbers close to 2^32,
  // but it probably isn't going to matter (and gives us a big performance win).
  if ((num >> 32) == 0) {
    return Sqrt32((uint32_t)num);
  }
  uint64_t res = 0;
  int max_bit_number = 64 - MostSignificantBit64(num);
  max_bit_number |= 1;
  uint64_t bit = 1ULL << (63 - max_bit_number);
  int iterations = (63 - max_bit_number) / 2 + 1;
  while (iterations--) {
    if (num >= res + bit) {
      num -= res + bit;
      res = (res >> 1U) + bit;
    } else {
      res >>= 1U;
    }
    bit >>= 2U;
  }
  // Do rounding - if we have the bits.
  if (num > res && res != 0xFFFFFFFFLL) {
    ++res;
  }
  return res;
}

uint32_t* FilterbankSqrt(struct FilterbankState* state, int scale_down_shift) {
  const int num_channels = state->num_channels;
  const uint64_t* work = state->work + 1;
  // Reuse the work buffer since we're fine clobbering it at this point to hold
  // the output.
  uint32_t* output = (uint32_t*)state->work;
  int i;
  for (i = 0; i < num_channels; ++i) {
    *output++ = Sqrt64(*work++) >> scale_down_shift;
  }
  return (uint32_t*)state->work;
}

void FilterbankReset(struct FilterbankState* state) {
  memset(state->work, 0, (state->num_channels + 1) * sizeof(*state->work));
}


void FftCompute(struct FftState* state, const int16_t* input,
                int input_scale_shift) {
  const size_t input_size = state->input_size;
  const size_t fft_size = state->fft_size;

  int16_t* fft_input = state->input;
  // First, scale the input by the given shift.
  size_t i;
  for (i = 0; i < input_size; ++i) {
    fft_input[i] = static_cast<int16_t>(static_cast<uint16_t>(input[i])
                                        << input_scale_shift);
  }
  // Zero out whatever else remains in the top part of the input.
  for (; i < fft_size; ++i) {
    fft_input[i] = 0;
  }

  // Apply the FFT.
  uint64_t t0_ = rdcycle64();
  kissfft_fixed16::kiss_fftr(
    reinterpret_cast<kissfft_fixed16::kiss_fftr_cfg>(state->scratch),
    state->input,
    reinterpret_cast<kissfft_fixed16::kiss_fft_cpx*>(state->output));
  uint64_t t1_ = rdcycle64();
  uint64_t t01_ = t1_ - t0_;
  printf("t(kissfft_fixed16::kiss_fftr)=%llu\n", t01_);
}

void FftInit(struct FftState* state) {
  // All the initialization is done in FftPopulateState()
}

void FftReset(struct FftState* state) {
  memset(state->input, 0, state->fft_size * sizeof(*state->input));
  memset(state->output, 0, (state->fft_size / 2 + 1) * sizeof(*state->output));
}

// Some platforms don't have M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void WindowFillConfigWithDefaults(struct WindowConfig* config) {
  config->size_ms = 25;
  config->step_size_ms = 10;
}

int WindowPopulateState(const struct WindowConfig* config,
                        struct WindowState* state, int sample_rate) {
  state->size = config->size_ms * sample_rate / 1000;
  state->step = config->step_size_ms * sample_rate / 1000;

  state->coefficients = (int16_t*) malloc(state->size * sizeof(*state->coefficients));
  if (state->coefficients == NULL) {
    printf("Failed to allocate window coefficients\n");
    return 0;
  }

  // Populate the window values.
  const float arg = M_PI * 2.0 / ((float)state->size);
  int i;
  for (i = 0; i < state->size; ++i) {
    float float_value = 0.5 - (0.5 * cos(arg * (i + 0.5)));
    // Scale it to fixed point and round it.
    state->coefficients[i] =
        floor(float_value * (1 << kFrontendWindowBits) + 0.5);
  }

  state->input_used = 0;
  state->input = (int16_t*)malloc(state->size * sizeof(*state->input));
  if (state->input == NULL) {
    printf("Failed to allocate window input\n");
    return 0;
  }

  state->output = (int16_t*)malloc(state->size * sizeof(*state->output));
  if (state->output == NULL) {
    printf("Failed to allocate window output\n");
    return 0;
  }

  return 1;
}

void WindowFreeStateContents(struct WindowState* state) {
  free(state->coefficients);
  free(state->input);
  free(state->output);
}

// #include "microfrontend/lib/bits.h"

struct FrontendOutput FrontendProcessSamples(struct FrontendState* state,
                                             const int16_t* samples,
                                             size_t num_samples,
                                             size_t* num_samples_read) {
  struct FrontendOutput output;
  output.values = NULL;
  output.size = 0;

  uint64_t t0 = rdcycle64();
  // Try to apply the window - if it fails, return and wait for more data.
  if (!WindowProcessSamples(&state->window, samples, num_samples,
                            num_samples_read)) {
    return output;
  }
  uint64_t t1 = rdcycle64();
  uint64_t t01 = t1 - t0;
  printf("t(WindowProcessSamples)=%llu\n", t01);

  uint64_t t0_ = rdcycle64();
  // Apply the FFT to the window's output (and scale it so that the fixed point
  // FFT can have as much resolution as possible).
  int input_shift =
      15 - MostSignificantBit32(state->window.max_abs_output_value);
  FftCompute(&state->fft, state->window.output, input_shift);
  uint64_t t1_ = rdcycle64();
  uint64_t t01_ = t1_ - t0_;
  printf("t(FftCompute)=%llu\n", t01_);

  // We can re-ruse the fft's output buffer to hold the energy.
  int32_t* energy = (int32_t*)state->fft.output;

  FilterbankConvertFftComplexToEnergy(&state->filterbank, state->fft.output,
                                      energy);

  FilterbankAccumulateChannels(&state->filterbank, energy);
  uint32_t* scaled_filterbank = FilterbankSqrt(&state->filterbank, input_shift);

  // Apply noise reduction.
  NoiseReductionApply(&state->noise_reduction, scaled_filterbank);

  if (state->pcan_gain_control.enable_pcan) {
    PcanGainControlApply(&state->pcan_gain_control, scaled_filterbank);
  }

  // Apply the log and scale.
  int correction_bits =
      MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
  uint16_t* logged_filterbank =
      LogScaleApply(&state->log_scale, scaled_filterbank,
                    state->filterbank.num_channels, correction_bits);

  output.size = state->filterbank.num_channels;
  output.values = logged_filterbank;
  return output;
}

void NoiseReductionFillConfigWithDefaults(struct NoiseReductionConfig* config) {
  config->smoothing_bits = 10;
  config->even_smoothing = 0.025;
  config->odd_smoothing = 0.06;
  config->min_signal_remaining = 0.05;
}

int NoiseReductionPopulateState(const struct NoiseReductionConfig* config,
                                struct NoiseReductionState* state,
                                int num_channels) {
  state->smoothing_bits = config->smoothing_bits;
  state->odd_smoothing = config->odd_smoothing * (1 << kNoiseReductionBits);
  state->even_smoothing = config->even_smoothing * (1 << kNoiseReductionBits);
  state->min_signal_remaining =
      config->min_signal_remaining * (1 << kNoiseReductionBits);
  state->num_channels = num_channels;
  state->estimate = (uint32_t*) calloc(state->num_channels, sizeof(*state->estimate));
  if (state->estimate == NULL) {
    printf("Failed to alloc estimate buffer\n");
    return 0;
  }
  return 1;
}

void NoiseReductionFreeStateContents(struct NoiseReductionState* state) {
  free(state->estimate);
}

#define kint16max 0x00007FFF

void PcanGainControlFillConfigWithDefaults(
    struct PcanGainControlConfig* config) {
  config->enable_pcan = 0;
  config->strength = 0.95;
  config->offset = 80.0;
  config->gain_bits = 21;
}

int16_t PcanGainLookupFunction(const struct PcanGainControlConfig* config,
                               int32_t input_bits, uint32_t x) {
  const float x_as_float = ((float)x) / ((uint32_t)1 << input_bits);
  const float gain_as_float =
      ((uint32_t)1 << config->gain_bits) *
      powf(x_as_float + config->offset, -config->strength);

  if (gain_as_float > kint16max) {
    return kint16max;
  }
  return (int16_t)(gain_as_float + 0.5f);
}

int PcanGainControlPopulateState(const struct PcanGainControlConfig* config,
                                 struct PcanGainControlState* state,
                                 uint32_t* noise_estimate,
                                 const int num_channels,
                                 const uint16_t smoothing_bits,
                                 const int32_t input_correction_bits) {
  state->enable_pcan = config->enable_pcan;
  if (!state->enable_pcan) {
    return 1;
  }
  state->noise_estimate = noise_estimate;
  state->num_channels = num_channels;
  state->gain_lut = (int16_t*) malloc(kWideDynamicFunctionLUTSize * sizeof(int16_t));
  if (state->gain_lut == NULL) {
    printf("Failed to allocate gain LUT\n");
    return 0;
  }
  state->snr_shift = config->gain_bits - input_correction_bits - kPcanSnrBits;

  const int32_t input_bits = smoothing_bits - input_correction_bits;
  state->gain_lut[0] = PcanGainLookupFunction(config, input_bits, 0);
  state->gain_lut[1] = PcanGainLookupFunction(config, input_bits, 1);
  state->gain_lut -= 6;
  int interval;
  for (interval = 2; interval <= kWideDynamicFunctionBits; ++interval) {
    const uint32_t x0 = (uint32_t)1 << (interval - 1);
    const uint32_t x1 = x0 + (x0 >> 1);
    const uint32_t x2 =
        (interval == kWideDynamicFunctionBits) ? x0 + (x0 - 1) : 2 * x0;

    const int16_t y0 = PcanGainLookupFunction(config, input_bits, x0);
    const int16_t y1 = PcanGainLookupFunction(config, input_bits, x1);
    const int16_t y2 = PcanGainLookupFunction(config, input_bits, x2);

    const int32_t diff1 = (int32_t)y1 - y0;
    const int32_t diff2 = (int32_t)y2 - y0;
    const int32_t a1 = 4 * diff1 - diff2;
    const int32_t a2 = diff2 - a1;

    state->gain_lut[4 * interval] = y0;
    state->gain_lut[4 * interval + 1] = (int16_t)a1;
    state->gain_lut[4 * interval + 2] = (int16_t)a2;
  }
  state->gain_lut += 6;
  return 1;
}

void PcanGainControlFreeStateContents(struct PcanGainControlState* state) {
  free(state->gain_lut);
}

void LogScaleFillConfigWithDefaults(struct LogScaleConfig* config) {
  config->enable_log = 1;
  config->scale_shift = 6;
}

int LogScalePopulateState(const struct LogScaleConfig* config,
                          struct LogScaleState* state) {
  state->enable_log = config->enable_log;
  state->scale_shift = config->scale_shift;
  return 1;
}

void FrontendReset(struct FrontendState* state) {
  WindowReset(&state->window);
  FftReset(&state->fft);
  FilterbankReset(&state->filterbank);
  NoiseReductionReset(&state->noise_reduction);
}

void FrontendFillConfigWithDefaults(struct FrontendConfig* config) {
  WindowFillConfigWithDefaults(&config->window);
  FilterbankFillConfigWithDefaults(&config->filterbank);
  NoiseReductionFillConfigWithDefaults(&config->noise_reduction);
  PcanGainControlFillConfigWithDefaults(&config->pcan_gain_control);
  LogScaleFillConfigWithDefaults(&config->log_scale);
}

int FrontendPopulateState(const struct FrontendConfig* config,
                          struct FrontendState* state, int sample_rate) {
  memset(state, 0, sizeof(*state));

  if (!WindowPopulateState(&config->window, &state->window, sample_rate)) {
    printf("Failed to populate window state\n");
    return 0;
  }

  if (!FftPopulateState(&state->fft, state->window.size)) {
    printf("Failed to populate fft state\n");
    return 0;
  }
  FftInit(&state->fft);

  if (!FilterbankPopulateState(&config->filterbank, &state->filterbank,
                               sample_rate, state->fft.fft_size / 2 + 1)) {
    printf("Failed to populate filterbank state\n");
    return 0;
  }

  if (!NoiseReductionPopulateState(&config->noise_reduction,
                                   &state->noise_reduction,
                                   state->filterbank.num_channels)) {
    printf("Failed to populate noise reduction state\n");
    return 0;
  }

  int input_correction_bits =
      MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
  if (!PcanGainControlPopulateState(
          &config->pcan_gain_control, &state->pcan_gain_control,
          state->noise_reduction.estimate, state->filterbank.num_channels,
          state->noise_reduction.smoothing_bits, input_correction_bits)) {
    printf("Failed to populate pcan gain control state\n");
    return 0;
  }

  if (!LogScalePopulateState(&config->log_scale, &state->log_scale)) {
    printf("Failed to populate log scale state\n");
    return 0;
  }

  FrontendReset(state);

  // All good, return a true value.
  return 1;
}

void FrontendFreeStateContents(struct FrontendState* state) {
  WindowFreeStateContents(&state->window);
  FftFreeStateContents(&state->fft);
  FilterbankFreeStateContents(&state->filterbank);
  NoiseReductionFreeStateContents(&state->noise_reduction);
  PcanGainControlFreeStateContents(&state->pcan_gain_control);
}

#ifndef FRONTEND_H
#define FRONTEND_H

#include <cstdint>

// Sets up any resources needed for the feature generation pipeline.
int InitializeFrontend();

// Converts audio sample data into a more compact form that's appropriate for
// feeding into a neural network.
int GenerateFrontendData(const int16_t* input, size_t input_size,
                               int8_t* output);

#endif  // FRONTEND_H

#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef MODEL_SETTINGS_H
#define MODEL_SETTINGS_H

#include <cstdint>

// #include "sdkconfig.h"

// The size of the input time series data we pass to the FFT to produce the
// frequency information. This has to be a power of two, and since we're dealing
// with 30ms of 16KHz inputs, which means 480 samples, this is the next larger
// value, i.e. 512.
constexpr int32_t max_audio_sample_size = 512;
constexpr int32_t audio_sample_frequency = 16000;

// The feature (powerspectrum image) on which the convolutional neural network
// operates on has 49 slices, each containing 40 grayscale pixels. So basically
// a 49 by 40 grayscale picture.
constexpr int32_t feature_slice_size = 40;
constexpr int32_t feature_slize_count = 49;
constexpr int32_t feature_element_count =
   feature_slice_size * feature_slize_count;

constexpr int32_t feature_slice_stride_ms = 20;
constexpr int32_t feature_slice_duration_ms = 30;

constexpr int32_t category_count = 10;
// extern const char* category_labels[category_count];

#endif  // MODEL_SETTINGS_H

FrontendState micro_features_state;

int InitializeFrontend() {
  // TODO(fabianpedd): Understand each value and try to finetune it. Refer to
  // the paper: "TRAINABLE FRONTEND FOR ROBUST AND FAR-FIELD KEYWORD SPOTTING"
  // TODO(fabianpedd): Check if the training in Keras, where the frontend is
  // part of the model, is changing/optimizing these parameters during training
  printf("000\n");
  FrontendConfig config;
  config.window.size_ms = feature_slice_duration_ms;
  config.window.step_size_ms = feature_slice_stride_ms;
  config.filterbank.num_channels = feature_slice_size;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  printf("111\n");

  if (!FrontendPopulateState(&config, &micro_features_state,
                             audio_sample_frequency)) {
    printf("ERROR: FrontendPopulateState() failed.");
    return -1;
  }
  printf("999\n");
  return 0;
}

int GenerateFrontendData(const int16_t* input, size_t input_size,
                               int8_t* output) {
  // TODO(fabianpedd): Simply add the 160 directly to input without the need for
  // an extra variable. But was is this code here for anyways!?!
  const int16_t* frontend_input;
  static bool is_first_time = true;
  if (is_first_time) {
    frontend_input = input;
    is_first_time = false;
  } else {
    // TODO(fabianpedd): Why do we need to add 160 here? Has this smth todo with
    // the comment from petewarden in feature_provider?
    frontend_input = input + 160;
  }

  // TODO(fabianpedd): Unused...
  size_t num_samples_read = 0;
  (void)num_samples_read;

  uint64_t t0 = rdcycle64();
  FrontendOutput frontend_output = FrontendProcessSamples(
      &micro_features_state, frontend_input, input_size, &num_samples_read);
  uint64_t t1 = rdcycle64();
  uint64_t t01 = t1 - t0;
  printf("t(FrontendProcessSamples)=%llu\n", t01);

  if (frontend_output.size <= 0 || frontend_output.values == NULL) {
    printf("ERROR: In FrontendProcessSamples().");
    return -1;
  }

  for (size_t i = 0; i < frontend_output.size; ++i) {
    // TODO(fabianpedd): Figure out what exactly is happening here with the
    // rounding and if this is related to the performance discrepancies between
    // training and inference.

    // These scaling values are derived from those used in
    // input_data.py in the training pipeline. The feature pipeline outputs
    // 16-bit signed integers in roughly a 0 to 670 range. In training, these
    // are then arbitrarily divided by 25.6 to get float values in the rough
    // range of 0.0 to 26.0. This scaling is performed for historical reasons,
    // to match up with the output of other feature generators. The process is
    // then further complicated when we quantize the model. This means we have
    // to scale the 0.0 to 26.0 real values to the -128 to 127 signed integer
    // numbers. All this means that to get matching values from our integer
    // feature output into the tensor input, we have to perform: input =
    // (((feature / 25.6) / 26.0) * 256) - 128 To simplify this and perform it
    // in 32-bit integer math, we rearrange to: input = (feature * 256) / (25.6
    // * 26.0) - 128
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
    int32_t value =
        ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
        value_div;
    value -= 128;
    if (value < -128) {
      value = -128;
    }
    if (value > 127) {
      value = 127;
    }
    output[i] = value;
  }

  return 0;
}

int bench_microfrontend() {
  printf("Hello World!\n");
  if (InitializeFrontend() != 0) {
    printf("ERROR: In InitializeFrontend().\n");
    return -1;
  }
  int8_t audio_buffer[960] = {0};

  // Contains our features. Interpreted as 40 by 49 byte 2d array.
  int8_t feature_buffer[feature_element_count];

  // Endless loop of main function.
  printf("Starting system main loop...\n");

  // while (true) {
  if (true) {
    // Get audio data from audio input and create slices until no more data is
    // available. But at most `feature_slize_count` times, which is equal to
    // 960ms of data. If we would
    uint64_t tq = rdcycle64();
    for (size_t i = 0; i < feature_slize_count; i++) {
      uint64_t ta = rdcycle64();
      // Get audio data via I2S from the audio driver.
      size_t actual_bytes_read = 0;
      int8_t i2s_read_buffer[640] = {0};
      actual_bytes_read = 640;
      // if (GetAudioData(640, &actual_bytes_read, i2s_read_buffer) != ESP_OK) {
      //   printf("ERROR: In GetAudioData().");
      //   return;
      // }

      // If there is no more audio data available at the moment, exit the
      // loop and continue with inference.
      if (actual_bytes_read < 640) {
        break;
      }

      // If there is a full 20ms / 320 samples / 640 bytes available, move the
      // old data (10ms / 160 samples / 320 bytes) to the top of the buffer and
      // fill the rest with the new audio data from the i2s_read_buffer.
      memcpy(audio_buffer, audio_buffer + 640, 320);
      memcpy(audio_buffer, i2s_read_buffer, 640);

      // Generate a new feature slice from the audio samples using the
      // GenerateFrontendData() function. This will convert the time domain
      // audio samples into a frequency domain representation.
      uint64_t t0 = rdcycle64();
      int8_t new_slice_buffer[feature_slice_size] = {0};
      if (GenerateFrontendData((int16_t*)audio_buffer, 512, new_slice_buffer) !=
          0) {
        printf("ERROR: In GenerateFrontendData().\n");
        return -1;
      }
      uint64_t t1 = rdcycle64();
      uint64_t t01 = t1 - t0;
      printf("t01=%llu\n", t01);
      // Move other slices by one, i.e. make room to store new slice at the end.
      // TODO(fabianpedd): This is actually really inefficient. Using a
      // ringbuffer would be a lot more efficient but also more
      // complicated. The ringbuffer could be used in such a way as to
      // minimize the amount of copying required. It would only be
      // necessary to copy the data when a ringbuffer overflow occurs. The
      // ringbuffer, in turn, would have to be at least 2-3x times the size of
      // the data we are expecting in order for this method to bring any
      // improvement. So we are basically trading in storage (of which we should
      // have plenty) for computations. But then again, how expensive are a
      // couple of memmoves and memcpys in the grand scheme of things here?
      memmove(feature_buffer, feature_buffer + feature_slice_size,
              feature_element_count - feature_slice_size);
      memcpy(feature_buffer + feature_element_count - feature_slice_size,
             new_slice_buffer, feature_slice_size);
      uint64_t tb = rdcycle64();
      uint64_t tab = tb - ta;
      printf("tab=%llu\n", tab);
    }
    uint64_t tw = rdcycle64();
    uint64_t tqw = tw - tq;
    printf("tqw=%llu\n", tqw);
  }
  return 0;
}
