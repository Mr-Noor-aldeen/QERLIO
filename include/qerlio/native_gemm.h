#ifndef QERLIO_NATIVE_GEMM_H
#define QERLIO_NATIVE_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && (defined(QERLIO_BUILD_SHARED) || defined(QERLIO_FREESTANDING))
#define QERLIO_GEMM_API __declspec(dllexport)
#elif defined(_WIN32) && defined(QERLIO_USE_SHARED)
#define QERLIO_GEMM_API __declspec(dllimport)
#else
#define QERLIO_GEMM_API
#endif

/**
 * Returns 1 if compiled with AVX2 + FMA + F16C SIMD support, 0 for portable scalar fallback.
 */
QERLIO_GEMM_API int qerlio_gemm_has_avx2(void);

/**
 * Compute y[t, r] = dot(x[t, :], dequant_q4_k(w_rows[r - row_start, :]))
 * for t in [0, m), r in [row_start, row_end).
 *
 * Parameters:
 *   x         : contiguous float32 row-major matrix of shape (m, k)
 *   w_rows    : contiguous Q4_K bytes for (row_end - row_start) rows, each row (k / 256) * 144 bytes
 *   y         : output float32 row-major matrix of shape (m, n)
 *   m         : number of activation rows (> 0)
 *   n         : total output columns / weight rows (> 0)
 *   k         : inner dimension (> 0, divisible by 256)
 *   row_start : inclusive start column index in y [0 .. n)
 *   row_end   : exclusive end column index in y [row_start .. n]
 *
 * Returns 0 on success, negative status code on invalid arguments.
 */
QERLIO_GEMM_API int qerlio_gemm_q4_k_f32(
    const float* x,
    const uint8_t* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
);

/**
 * Same contract as qerlio_gemm_q4_k_f32, for GGML Q6_K weights ((k / 256) * 210 bytes per row).
 */
QERLIO_GEMM_API int qerlio_gemm_q6_k_f32(
    const float* x,
    const uint8_t* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
);

/**
 * Same contract as qerlio_gemm_q4_k_f32, for GGML Q8_0 weights ((k / 32) * 34 bytes per row).
 */
QERLIO_GEMM_API int qerlio_gemm_q8_0_f32(
    const float* x,
    const uint8_t* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
);

/**
 * Same contract as qerlio_gemm_q4_k_f32, for uncompressed float32 weights (k floats per row).
 */
QERLIO_GEMM_API int qerlio_gemm_f32(
    const float* x,
    const float* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
);

#ifdef __cplusplus
}
#endif

#endif /* QERLIO_NATIVE_GEMM_H */
