#include "qerlio/native_gemm.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define QERLIO_USE_AVX2 1
#include <immintrin.h>
#else
#define QERLIO_USE_AVX2 0
#endif

#if defined(_WIN32) && defined(QERLIO_FREESTANDING)
/* Satisfies MSVC/LLVM floating-point CRT sentinel when linked with /NOENTRY */
int _fltused = 0;
#endif

static uint16_t read_le_u16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int32_t byte_to_i8(uint8_t b) {
    return (b < 128u) ? (int32_t)b : ((int32_t)b - 256);
}

static float fp16_to_f32(uint16_t h) {
#if QERLIO_USE_AVX2
    return _cvtsh_ss(h);
#else
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    uint32_t bits = 0;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            /* Subnormal binary16 -> normalize into binary32 */
            exp = 127u - 14u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                exp -= 1u;
            }
            mant &= 0x03FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = bits;
    return conv.f;
#endif
}

static int validate_gemm_args(
    const void* x,
    const void* w_rows,
    const float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end,
    int64_t k_align
) {
    if (!x || !w_rows || !y) {
        return -1;
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return -2;
    }
    if (row_start < 0 || row_end < row_start || row_end > n) {
        return -3;
    }
    if (k_align > 1 && (k % k_align) != 0) {
        return -4;
    }
    return 0;
}

#if QERLIO_USE_AVX2
static float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 shuf2 = _mm_movehl_ps(sum2, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf2);
    return _mm_cvtss_f32(sum1);
}
#endif

QERLIO_GEMM_API int qerlio_gemm_has_avx2(void) {
    return QERLIO_USE_AVX2;
}

/* ------------------------------------------------------------------------- */
/* Q4_K GEMM kernel (144 bytes per 256 weights)                              */
/* ------------------------------------------------------------------------- */

#if QERLIO_USE_AVX2
static void gemm_q4_k_row_tile4_avx2(
    const float* x0,
    const float* x1,
    const float* x2,
    const float* x3,
    int64_t mt,
    const uint8_t* w_row,
    int64_t n_blocks,
    float* out_acc
) {
    const __m256i mask0f = _mm256_set1_epi8(0x0F);

    out_acc[0] = 0.0f;
    out_acc[1] = 0.0f;
    out_acc[2] = 0.0f;
    out_acc[3] = 0.0f;

    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = w_row + b * 144;
        const float d = fp16_to_f32(read_le_u16(blk));
        const float dmin = fp16_to_f32(read_le_u16(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;

        uint8_t sc[8];
        uint8_t mn[8];
        for (int j = 0; j < 4; ++j) {
            sc[j] = scales[j] & 63u;
            mn[j] = scales[j + 4] & 63u;
        }
        for (int j = 4; j < 8; ++j) {
            sc[j] = (uint8_t)((scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4));
            mn[j] = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }

        const int64_t k_base = b * 256;
        for (int c = 0; c < 4; ++c) {
            const float d1 = d * (float)sc[2 * c];
            const float m1 = dmin * (float)mn[2 * c];
            const float d2 = d * (float)sc[2 * c + 1];
            const float m2 = dmin * (float)mn[2 * c + 1];

            const __m256i qv = _mm256_loadu_si256((const __m256i*)(qs + c * 32));
            const __m256i q_lo = _mm256_and_si256(qv, mask0f);
            const __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), mask0f);

            const __m128i lo0 = _mm256_castsi256_si128(q_lo);
            const __m128i lo1 = _mm256_extracti128_si256(q_lo, 1);
            const __m128i hi0 = _mm256_castsi256_si128(q_hi);
            const __m128i hi1 = _mm256_extracti128_si256(q_hi, 1);

            const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo0));
            const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo0, 8)));
            const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo1));
            const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo1, 8)));

            const __m256 w4 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi0));
            const __m256 w5 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi0, 8)));
            const __m256 w6 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi1));
            const __m256 w7 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi1, 8)));

            const int64_t off = k_base + (int64_t)c * 64;
            const float* xp[4] = {x0 + off, x1 + off, x2 + off, x3 + off};

            for (int64_t t = 0; t < mt; ++t) {
                const float* p = xp[t];

                __m256 xv0 = _mm256_loadu_ps(p + 0);
                __m256 xv1 = _mm256_loadu_ps(p + 8);
                __m256 xv2 = _mm256_loadu_ps(p + 16);
                __m256 xv3 = _mm256_loadu_ps(p + 24);
                __m256 dot_lo = _mm256_mul_ps(w0, xv0);
                __m256 sum_lo = _mm256_add_ps(xv0, xv1);
                dot_lo = _mm256_fmadd_ps(w1, xv1, dot_lo);
                sum_lo = _mm256_add_ps(sum_lo, xv2);
                dot_lo = _mm256_fmadd_ps(w2, xv2, dot_lo);
                sum_lo = _mm256_add_ps(sum_lo, xv3);
                dot_lo = _mm256_fmadd_ps(w3, xv3, dot_lo);

                __m256 xv4 = _mm256_loadu_ps(p + 32);
                __m256 xv5 = _mm256_loadu_ps(p + 40);
                __m256 xv6 = _mm256_loadu_ps(p + 48);
                __m256 xv7 = _mm256_loadu_ps(p + 56);
                __m256 dot_hi = _mm256_mul_ps(w4, xv4);
                __m256 sum_hi = _mm256_add_ps(xv4, xv5);
                dot_hi = _mm256_fmadd_ps(w5, xv5, dot_hi);
                sum_hi = _mm256_add_ps(sum_hi, xv6);
                dot_hi = _mm256_fmadd_ps(w6, xv6, dot_hi);
                sum_hi = _mm256_add_ps(sum_hi, xv7);
                dot_hi = _mm256_fmadd_ps(w7, xv7, dot_hi);

                out_acc[t] += (d1 * hsum256_ps(dot_lo) - m1 * hsum256_ps(sum_lo))
                            + (d2 * hsum256_ps(dot_hi) - m2 * hsum256_ps(sum_hi));
            }
        }
    }
}
#else
static void gemm_q4_k_row_scalar(
    const float* x_row,
    const uint8_t* w_row,
    int64_t n_blocks,
    float* out_val
) {
    float acc = 0.0f;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = w_row + b * 144;
        const float d = fp16_to_f32(read_le_u16(blk));
        const float dmin = fp16_to_f32(read_le_u16(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;

        uint8_t sc[8];
        uint8_t mn[8];
        for (int j = 0; j < 4; ++j) {
            sc[j] = scales[j] & 63u;
            mn[j] = scales[j + 4] & 63u;
        }
        for (int j = 4; j < 8; ++j) {
            sc[j] = (uint8_t)((scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4));
            mn[j] = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }

        const float* xb = x_row + b * 256;
        for (int c = 0; c < 4; ++c) {
            const float d1 = d * (float)sc[2 * c];
            const float m1 = dmin * (float)mn[2 * c];
            const float d2 = d * (float)sc[2 * c + 1];
            const float m2 = dmin * (float)mn[2 * c + 1];
            const uint8_t* qc = qs + c * 32;
            const float* xc = xb + c * 64;
            float dot_lo = 0.0f, sum_lo = 0.0f;
            float dot_hi = 0.0f, sum_hi = 0.0f;
            for (int l = 0; l < 32; ++l) {
                const float q_lo = (float)(qc[l] & 0x0Fu);
                const float q_hi = (float)(qc[l] >> 4);
                const float xl = xc[l];
                const float xh = xc[32 + l];
                dot_lo += q_lo * xl;
                sum_lo += xl;
                dot_hi += q_hi * xh;
                sum_hi += xh;
            }
            acc += (d1 * dot_lo - m1 * sum_lo) + (d2 * dot_hi - m2 * sum_hi);
        }
    }
    *out_val = acc;
}
#endif

QERLIO_GEMM_API int qerlio_gemm_q4_k_f32(
    const float* x,
    const uint8_t* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
) {
    const int rc = validate_gemm_args(x, w_rows, y, m, n, k, row_start, row_end, 256);
    if (rc != 0) {
        return rc;
    }

    const int64_t n_blocks = k / 256;
    const int64_t row_bytes = n_blocks * 144;
    const int64_t local_rows = row_end - row_start;

    for (int64_t lr = 0; lr < local_rows; ++lr) {
        const uint8_t* w_row = w_rows + lr * row_bytes;
        const int64_t r = row_start + lr;

#if QERLIO_USE_AVX2
        for (int64_t t0 = 0; t0 < m; t0 += 4) {
            const int64_t mt = (m - t0 < 4) ? (m - t0) : 4;
            const float* x0 = x + t0 * k;
            const float* x1 = (mt > 1) ? (x0 + k) : x0;
            const float* x2 = (mt > 2) ? (x0 + 2 * k) : x0;
            const float* x3 = (mt > 3) ? (x0 + 3 * k) : x0;
            float acc[4];
            gemm_q4_k_row_tile4_avx2(x0, x1, x2, x3, mt, w_row, n_blocks, acc);
            for (int64_t dt = 0; dt < mt; ++dt) {
                y[(t0 + dt) * n + r] = acc[dt];
            }
        }
#else
        for (int64_t t = 0; t < m; ++t) {
            float val = 0.0f;
            gemm_q4_k_row_scalar(x + t * k, w_row, n_blocks, &val);
            y[t * n + r] = val;
        }
#endif
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Q6_K GEMM kernel (210 bytes per 256 weights)                              */
/* ------------------------------------------------------------------------- */

#if QERLIO_USE_AVX2
static void gemm_q6_k_row_tile4_avx2(
    const float* x0,
    const float* x1,
    const float* x2,
    const float* x3,
    int64_t mt,
    const uint8_t* w_row,
    int64_t n_blocks,
    float* out_acc
) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i m03 = _mm_set1_epi8(0x03);
    const __m256i sub32 = _mm256_set1_epi32(32);

    out_acc[0] = 0.0f;
    out_acc[1] = 0.0f;
    out_acc[2] = 0.0f;
    out_acc[3] = 0.0f;

    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = w_row + b * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const uint8_t* scales = blk + 192;
        const float d = fp16_to_f32(read_le_u16(blk + 208));

        for (int n_half = 0; n_half < 2; ++n_half) {
            const uint8_t* ql_0 = ql + n_half * 64;
            const uint8_t* ql_1 = ql + n_half * 64 + 32;
            const uint8_t* qh_h = qh + n_half * 32;
            const uint8_t* sc_h = scales + n_half * 8;
            const int64_t k_half = b * 256 + n_half * 128;

            for (int is_sub = 0; is_sub < 2; ++is_sub) {
                const int l_off = is_sub * 16;
                const float s1 = d * (float)byte_to_i8(sc_h[is_sub + 0]);
                const float s2 = d * (float)byte_to_i8(sc_h[is_sub + 2]);
                const float s3 = d * (float)byte_to_i8(sc_h[is_sub + 4]);
                const float s4 = d * (float)byte_to_i8(sc_h[is_sub + 6]);

                const __m128i v_ql0 = _mm_loadu_si128((const __m128i*)(ql_0 + l_off));
                const __m128i v_ql1 = _mm_loadu_si128((const __m128i*)(ql_1 + l_off));
                const __m128i v_qh = _mm_loadu_si128((const __m128i*)(qh_h + l_off));

                const __m128i q1_b = _mm_or_si128(
                    _mm_and_si128(v_ql0, m0f),
                    _mm_slli_epi16(_mm_and_si128(v_qh, m03), 4)
                );
                const __m128i q2_b = _mm_or_si128(
                    _mm_and_si128(v_ql1, m0f),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(v_qh, 2), m03), 4)
                );
                const __m128i q3_b = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(v_ql0, 4), m0f),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(v_qh, 4), m03), 4)
                );
                const __m128i q4_b = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(v_ql1, 4), m0f),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(v_qh, 6), m03), 4)
                );

                const __m256 w1_lo = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q1_b), sub32));
                const __m256 w1_hi = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q1_b, 8)), sub32));
                const __m256 w2_lo = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q2_b), sub32));
                const __m256 w2_hi = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q2_b, 8)), sub32));
                const __m256 w3_lo = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q3_b), sub32));
                const __m256 w3_hi = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q3_b, 8)), sub32));
                const __m256 w4_lo = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q4_b), sub32));
                const __m256 w4_hi = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q4_b, 8)), sub32));

                const int64_t off = k_half + l_off;
                const float* xp[4] = {x0 + off, x1 + off, x2 + off, x3 + off};

                for (int64_t t = 0; t < mt; ++t) {
                    const float* p = xp[t];

                    __m256 d1v = _mm256_mul_ps(w1_lo, _mm256_loadu_ps(p + 0));
                    d1v = _mm256_fmadd_ps(w1_hi, _mm256_loadu_ps(p + 8), d1v);

                    __m256 d2v = _mm256_mul_ps(w2_lo, _mm256_loadu_ps(p + 32));
                    d2v = _mm256_fmadd_ps(w2_hi, _mm256_loadu_ps(p + 40), d2v);

                    __m256 d3v = _mm256_mul_ps(w3_lo, _mm256_loadu_ps(p + 64));
                    d3v = _mm256_fmadd_ps(w3_hi, _mm256_loadu_ps(p + 72), d3v);

                    __m256 d4v = _mm256_mul_ps(w4_lo, _mm256_loadu_ps(p + 96));
                    d4v = _mm256_fmadd_ps(w4_hi, _mm256_loadu_ps(p + 104), d4v);

                    out_acc[t] += s1 * hsum256_ps(d1v)
                                + s2 * hsum256_ps(d2v)
                                + s3 * hsum256_ps(d3v)
                                + s4 * hsum256_ps(d4v);
                }
            }
        }
    }
}
#else
static void gemm_q6_k_row_scalar(
    const float* x_row,
    const uint8_t* w_row,
    int64_t n_blocks,
    float* out_val
) {
    float acc = 0.0f;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = w_row + b * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const uint8_t* scales = blk + 192;
        const float d = fp16_to_f32(read_le_u16(blk + 208));

        for (int n_half = 0; n_half < 2; ++n_half) {
            const uint8_t* ql_0 = ql + n_half * 64;
            const uint8_t* ql_1 = ql + n_half * 64 + 32;
            const uint8_t* qh_h = qh + n_half * 32;
            const uint8_t* sc_h = scales + n_half * 8;
            const float* xh = x_row + b * 256 + n_half * 128;

            for (int l = 0; l < 32; ++l) {
                const int is_sub = l >> 4;
                const float q1 = (float)((int32_t)((ql_0[l] & 0x0Fu) | (((qh_h[l] >> 0) & 3u) << 4)) - 32);
                const float q2 = (float)((int32_t)((ql_1[l] & 0x0Fu) | (((qh_h[l] >> 2) & 3u) << 4)) - 32);
                const float q3 = (float)((int32_t)((ql_0[l] >> 4) | (((qh_h[l] >> 4) & 3u) << 4)) - 32);
                const float q4 = (float)((int32_t)((ql_1[l] >> 4) | (((qh_h[l] >> 6) & 3u) << 4)) - 32);

                const float s1 = d * (float)byte_to_i8(sc_h[is_sub + 0]);
                const float s2 = d * (float)byte_to_i8(sc_h[is_sub + 2]);
                const float s3 = d * (float)byte_to_i8(sc_h[is_sub + 4]);
                const float s4 = d * (float)byte_to_i8(sc_h[is_sub + 6]);

                acc += (s1 * q1) * xh[l + 0]
                     + (s2 * q2) * xh[l + 32]
                     + (s3 * q3) * xh[l + 64]
                     + (s4 * q4) * xh[l + 96];
            }
        }
    }
    *out_val = acc;
}
#endif

QERLIO_GEMM_API int qerlio_gemm_q6_k_f32(
    const float* x,
    const uint8_t* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
) {
    const int rc = validate_gemm_args(x, w_rows, y, m, n, k, row_start, row_end, 256);
    if (rc != 0) {
        return rc;
    }

    const int64_t n_blocks = k / 256;
    const int64_t row_bytes = n_blocks * 210;
    const int64_t local_rows = row_end - row_start;

    for (int64_t lr = 0; lr < local_rows; ++lr) {
        const uint8_t* w_row = w_rows + lr * row_bytes;
        const int64_t r = row_start + lr;

#if QERLIO_USE_AVX2
        for (int64_t t0 = 0; t0 < m; t0 += 4) {
            const int64_t mt = (m - t0 < 4) ? (m - t0) : 4;
            const float* x0 = x + t0 * k;
            const float* x1 = (mt > 1) ? (x0 + k) : x0;
            const float* x2 = (mt > 2) ? (x0 + 2 * k) : x0;
            const float* x3 = (mt > 3) ? (x0 + 3 * k) : x0;
            float acc[4];
            gemm_q6_k_row_tile4_avx2(x0, x1, x2, x3, mt, w_row, n_blocks, acc);
            for (int64_t dt = 0; dt < mt; ++dt) {
                y[(t0 + dt) * n + r] = acc[dt];
            }
        }
#else
        for (int64_t t = 0; t < m; ++t) {
            float val = 0.0f;
            gemm_q6_k_row_scalar(x + t * k, w_row, n_blocks, &val);
            y[t * n + r] = val;
        }
#endif
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Q8_0 GEMM kernel (34 bytes per 32 weights)                                */
/* ------------------------------------------------------------------------- */

QERLIO_GEMM_API int qerlio_gemm_q8_0_f32(
    const float* x,
    const uint8_t* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
) {
    const int rc = validate_gemm_args(x, w_rows, y, m, n, k, row_start, row_end, 32);
    if (rc != 0) {
        return rc;
    }

    const int64_t n_blocks = k / 32;
    const int64_t row_bytes = n_blocks * 34;
    const int64_t local_rows = row_end - row_start;

    for (int64_t lr = 0; lr < local_rows; ++lr) {
        const uint8_t* w_row = w_rows + lr * row_bytes;
        const int64_t r = row_start + lr;

        for (int64_t t = 0; t < m; ++t) {
            const float* xr = x + t * k;
            float acc = 0.0f;
            for (int64_t b = 0; b < n_blocks; ++b) {
                const uint8_t* blk = w_row + b * 34;
                const float d = fp16_to_f32(read_le_u16(blk));
                const uint8_t* qs = blk + 2;
                const float* xb = xr + b * 32;
#if QERLIO_USE_AVX2
                const __m128i qlo = _mm_loadu_si128((const __m128i*)(qs + 0));
                const __m128i qhi = _mm_loadu_si128((const __m128i*)(qs + 16));
                const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qlo));
                const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qlo, 8)));
                const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qhi));
                const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qhi, 8)));

                __m256 dv = _mm256_mul_ps(w0, _mm256_loadu_ps(xb + 0));
                dv = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb + 8), dv);
                dv = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb + 16), dv);
                dv = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb + 24), dv);
                acc += d * hsum256_ps(dv);
#else
                float bdot = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    bdot += (float)byte_to_i8(qs[i]) * xb[i];
                }
                acc += d * bdot;
#endif
            }
            y[t * n + r] = acc;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* F32 GEMM kernel (k float32 elements per row)                              */
/* ------------------------------------------------------------------------- */

QERLIO_GEMM_API int qerlio_gemm_f32(
    const float* x,
    const float* w_rows,
    float* y,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t row_start,
    int64_t row_end
) {
    const int rc = validate_gemm_args(x, w_rows, y, m, n, k, row_start, row_end, 1);
    if (rc != 0) {
        return rc;
    }

    const int64_t local_rows = row_end - row_start;
    for (int64_t lr = 0; lr < local_rows; ++lr) {
        const float* w_row = w_rows + lr * k;
        const int64_t r = row_start + lr;
        for (int64_t t = 0; t < m; ++t) {
            const float* xr = x + t * k;
            float acc = 0.0f;
            int64_t col = 0;
#if QERLIO_USE_AVX2
            __m256 vacc = _mm256_setzero_ps();
            for (; col + 8 <= k; col += 8) {
                vacc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(xr + col),
                    _mm256_loadu_ps(w_row + col),
                    vacc
                );
            }
            acc = hsum256_ps(vacc);
#endif
            for (; col < k; ++col) {
                acc += xr[col] * w_row[col];
            }
            y[t * n + r] = acc;
        }
    }
    return 0;
}
