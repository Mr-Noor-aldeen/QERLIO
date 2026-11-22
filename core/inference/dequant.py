"""From-scratch GGML block dequantization kernels using only Python stdlib and NumPy.

Implements both vectorized NumPy dequantization (`dequantize_bytes`) and an independent
pure-Python scalar reference (`dequantize_block_scalar_reference`) for exact verification.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Tuple, Union

import numpy as np

# GGML tensor type IDs
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30

KVALUES_IQ4NL: Tuple[int, ...] = (
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
)
_KVALUES_IQ4NL_NP = np.array(KVALUES_IQ4NL, dtype=np.float32)

IQ3S_GRID_U32: Tuple[int, ...] = (
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
)
_IQ3S_GRID_NP = (
    np.array(IQ3S_GRID_U32, dtype="<u4")
    .view(np.uint8)
    .reshape(512, 4)
    .astype(np.float32)
)

GGML_TYPE_NAMES: Dict[int, str] = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0",
    GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q8_1: "Q8_1",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_Q8_K: "Q8_K",
    GGML_TYPE_IQ2_XXS: "IQ2_XXS",
    GGML_TYPE_IQ2_XS: "IQ2_XS",
    GGML_TYPE_IQ3_XXS: "IQ3_XXS",
    GGML_TYPE_IQ1_S: "IQ1_S",
    GGML_TYPE_IQ4_NL: "IQ4_NL",
    GGML_TYPE_IQ3_S: "IQ3_S",
    GGML_TYPE_IQ2_S: "IQ2_S",
    GGML_TYPE_IQ4_XS: "IQ4_XS",
    GGML_TYPE_I8: "I8",
    GGML_TYPE_I16: "I16",
    GGML_TYPE_I32: "I32",
    GGML_TYPE_I64: "I64",
    GGML_TYPE_F64: "F64",
    GGML_TYPE_IQ1_M: "IQ1_M",
    GGML_TYPE_BF16: "BF16",
}

# Mapping: type_id -> (block_size_elements, block_size_bytes)
GGML_TYPE_INFO: Dict[int, Tuple[int, int]] = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_Q4_0: (32, 18),
    GGML_TYPE_Q4_1: (32, 20),
    GGML_TYPE_Q5_0: (32, 22),
    GGML_TYPE_Q5_1: (32, 24),
    GGML_TYPE_Q8_0: (32, 34),
    GGML_TYPE_Q8_1: (32, 36),
    GGML_TYPE_Q2_K: (256, 84),
    GGML_TYPE_Q3_K: (256, 110),
    GGML_TYPE_Q4_K: (256, 144),
    GGML_TYPE_Q5_K: (256, 176),
    GGML_TYPE_Q6_K: (256, 210),
    GGML_TYPE_Q8_K: (256, 292),
    GGML_TYPE_IQ2_XXS: (256, 66),
    GGML_TYPE_IQ2_XS: (256, 74),
    GGML_TYPE_IQ3_XXS: (256, 98),
    GGML_TYPE_IQ1_S: (256, 50),
    GGML_TYPE_IQ4_NL: (32, 18),
    GGML_TYPE_IQ3_S: (256, 110),
    GGML_TYPE_IQ2_S: (256, 82),
    GGML_TYPE_IQ4_XS: (256, 136),
    GGML_TYPE_I8: (1, 1),
    GGML_TYPE_I16: (1, 2),
    GGML_TYPE_I32: (1, 4),
    GGML_TYPE_I64: (1, 8),
    GGML_TYPE_F64: (1, 8),
    GGML_TYPE_IQ1_M: (256, 56),
    GGML_TYPE_BF16: (1, 2),
}


def get_block_size(ggml_type: int) -> int:
    """Return number of elements per quantization block for the given GGML type."""
    if ggml_type not in GGML_TYPE_INFO:
        raise ValueError(f"Unsupported GGML type ID: {ggml_type}")
    return GGML_TYPE_INFO[ggml_type][0]


def get_type_size(ggml_type: int) -> int:
    """Return byte size per quantization block for the given GGML type."""
    if ggml_type not in GGML_TYPE_INFO:
        raise ValueError(f"Unsupported GGML type ID: {ggml_type}")
    return GGML_TYPE_INFO[ggml_type][1]


def compute_tensor_bytes(ggml_type: int, num_elements: int) -> int:
    """Compute exact byte length of a tensor with `num_elements` in `ggml_type`."""
    block_size, type_size = GGML_TYPE_INFO.get(ggml_type, (0, 0))
    if block_size == 0:
        raise ValueError(f"Unsupported GGML type ID: {ggml_type}")
    if num_elements % block_size != 0:
        raise ValueError(
            f"Element count {num_elements} is not divisible by block size {block_size} "
            f"for type {GGML_TYPE_NAMES.get(ggml_type, str(ggml_type))}"
        )
    return (num_elements // block_size) * type_size


def _fp16_bytes_to_f32_scalar(raw2: bytes) -> float:
    """Unpack a 2-byte IEEE 754 binary16 into a Python float via float32 precision."""
    val_f16 = struct.unpack("<e", raw2)[0]
    # Round-trip through IEEE 754 binary32 so scalar arithmetic matches float32 exactly
    return struct.unpack("<f", struct.pack("<f", val_f16))[0]


def _f32_round(val: float) -> float:
    """Round a Python float64 to single-precision IEEE 754 float32."""
    return struct.unpack("<f", struct.pack("<f", val))[0]


def dequantize_block_scalar_reference(
    block_bytes: Union[bytes, bytearray, memoryview],
    ggml_type: int,
) -> List[float]:
    """Pure-Python scalar reference implementation of single-block GGML dequantization.

    Follows the exact GGML block layout specification element-by-element using `struct`
    and IEEE-754 single-precision rounding so it can serve as an independent mathematical
    oracle for testing the vectorized NumPy kernels.
    """
    raw = bytes(block_bytes)
    block_size, type_size = GGML_TYPE_INFO.get(ggml_type, (0, 0))
    if block_size == 0:
        raise ValueError(f"Unsupported GGML type ID: {ggml_type}")
    if len(raw) != type_size:
        raise ValueError(
            f"Expected {type_size} bytes for single block of type "
            f"{GGML_TYPE_NAMES.get(ggml_type, str(ggml_type))}, got {len(raw)}"
        )

    if ggml_type == GGML_TYPE_F32:
        return [struct.unpack("<f", raw)[0]]

    if ggml_type == GGML_TYPE_F16:
        return [_fp16_bytes_to_f32_scalar(raw)]

    if ggml_type == GGML_TYPE_BF16:
        u16 = struct.unpack("<H", raw)[0]
        return [struct.unpack("<f", struct.pack("<I", u16 << 16))[0]]

    if ggml_type == GGML_TYPE_Q4_0:
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        qs = raw[2:18]
        low = [_f32_round(((b & 0x0F) - 8) * d) for b in qs]
        high = [_f32_round(((b >> 4) - 8) * d) for b in qs]
        return low + high

    if ggml_type == GGML_TYPE_Q4_1:
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        m = _fp16_bytes_to_f32_scalar(raw[2:4])
        qs = raw[4:20]
        low = [_f32_round(_f32_round((b & 0x0F) * d) + m) for b in qs]
        high = [_f32_round(_f32_round((b >> 4) * d) + m) for b in qs]
        return low + high

    if ggml_type == GGML_TYPE_Q5_0:
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        qh = struct.unpack("<I", raw[2:6])[0]
        qs = raw[6:22]
        low: List[float] = []
        high: List[float] = []
        for j in range(16):
            xh_0 = ((qh >> j) & 1) << 4
            xh_1 = ((qh >> (j + 16)) & 1) << 4
            q0 = ((qs[j] & 0x0F) | xh_0) - 16
            q1 = ((qs[j] >> 4) | xh_1) - 16
            low.append(_f32_round(q0 * d))
            high.append(_f32_round(q1 * d))
        return low + high

    if ggml_type == GGML_TYPE_Q5_1:
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        m = _fp16_bytes_to_f32_scalar(raw[2:4])
        qh = struct.unpack("<I", raw[4:8])[0]
        qs = raw[8:24]
        low = []
        high = []
        for j in range(16):
            xh_0 = ((qh >> j) & 1) << 4
            xh_1 = ((qh >> (j + 16)) & 1) << 4
            q0 = (qs[j] & 0x0F) | xh_0
            q1 = (qs[j] >> 4) | xh_1
            low.append(_f32_round(_f32_round(q0 * d) + m))
            high.append(_f32_round(_f32_round(q1 * d) + m))
        return low + high

    if ggml_type == GGML_TYPE_Q8_0:
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        qs_i8 = struct.unpack("<32b", raw[2:34])
        return [_f32_round(d * q) for q in qs_i8]

    if ggml_type == GGML_TYPE_Q2_K:
        # Layout (84 bytes):
        # scales: 16 uint8 bytes (offset 0..16) -> low 4 bits scale, high 4 bits min
        # qs: 64 uint8 bytes (offset 16..80) -> 256 2-bit values
        # d: fp16 (offset 80..82)
        # dmin: fp16 (offset 82..84)
        scales = raw[0:16]
        qs = raw[16:80]
        d = _fp16_bytes_to_f32_scalar(raw[80:82])
        dmin = _fp16_bytes_to_f32_scalar(raw[82:84])
        out: List[float] = []
        is_idx = 0
        for n in range(2):
            q_chunk = qs[n * 32 : (n + 1) * 32]
            for shift in (0, 2, 4, 6):
                sc0 = scales[is_idx]
                dl0 = _f32_round(d * (sc0 & 0x0F))
                ml0 = _f32_round(dmin * (sc0 >> 4))
                is_idx += 1
                for l in range(16):
                    q = (q_chunk[l] >> shift) & 3
                    out.append(_f32_round(_f32_round(dl0 * q) - ml0))

                sc1 = scales[is_idx]
                dl1 = _f32_round(d * (sc1 & 0x0F))
                ml1 = _f32_round(dmin * (sc1 >> 4))
                is_idx += 1
                for l in range(16, 32):
                    q = (q_chunk[l] >> shift) & 3
                    out.append(_f32_round(_f32_round(dl1 * q) - ml1))
        return out

    if ggml_type == GGML_TYPE_Q3_K:
        # Layout (110 bytes):
        # hmask: 32 uint8 bytes (offset 0..32)
        # qs: 64 uint8 bytes (offset 32..96)
        # scales: 12 uint8 bytes (offset 96..108) -> 16 6-bit signed scales (unpacked - 32)
        # d: fp16 (offset 108..110)
        hmask = raw[0:32]
        qs = raw[32:96]
        sc_raw = raw[96:108]
        d = _fp16_bytes_to_f32_scalar(raw[108:110])

        # Unpack 16 6-bit scales from 12 bytes matching GGML Q3_K layout
        sc = [0] * 16
        for i in range(8):
            sc[i] = sc_raw[i] & 0x0F
            sc[i + 8] = sc_raw[i] >> 4
        for i in range(4):
            b = sc_raw[8 + i]
            sc[i] |= (b & 0x03) << 4
            sc[i + 4] |= ((b >> 2) & 0x03) << 4
            sc[i + 8] |= ((b >> 4) & 0x03) << 4
            sc[i + 12] |= ((b >> 6) & 0x03) << 4
        scales_signed = [s - 32 for s in sc]

        out = []
        is_idx = 0
        m_bit = 1
        for n in range(2):
            q_chunk = qs[n * 32 : (n + 1) * 32]
            for shift in (0, 2, 4, 6):
                dl = _f32_round(d * scales_signed[is_idx])
                is_idx += 1
                for l in range(16):
                    q = ((q_chunk[l] >> shift) & 3) - (0 if (hmask[l] & m_bit) else 4)
                    out.append(_f32_round(dl * q))
                dl = _f32_round(d * scales_signed[is_idx])
                is_idx += 1
                for l in range(16, 32):
                    q = ((q_chunk[l] >> shift) & 3) - (0 if (hmask[l] & m_bit) else 4)
                    out.append(_f32_round(dl * q))
                m_bit <<= 1
        return out

    if ggml_type == GGML_TYPE_Q4_K:
        # Layout (144 bytes):
        # d: fp16 (0..2)
        # dmin: fp16 (2..4)
        # scales: 12 uint8 (4..16)
        # qs: 128 uint8 (16..144)
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        dmin = _fp16_bytes_to_f32_scalar(raw[2:4])
        scales = raw[4:16]
        qs = raw[16:144]

        sc = [0] * 8
        m = [0] * 8
        for j in range(4):
            sc[j] = scales[j] & 63
            m[j] = scales[j + 4] & 63
        for j in range(4, 8):
            sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
            m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)

        out = []
        for c in range(4):
            qs_chunk = qs[c * 32 : (c + 1) * 32]
            d1 = _f32_round(d * sc[2 * c])
            m1 = _f32_round(dmin * m[2 * c])
            d2 = _f32_round(d * sc[2 * c + 1])
            m2 = _f32_round(dmin * m[2 * c + 1])
            for l in range(32):
                q_low = qs_chunk[l] & 0x0F
                out.append(_f32_round(_f32_round(d1 * q_low) - m1))
            for l in range(32):
                q_high = qs_chunk[l] >> 4
                out.append(_f32_round(_f32_round(d2 * q_high) - m2))
        return out

    if ggml_type == GGML_TYPE_Q5_K:
        # Layout (176 bytes):
        # d: fp16 (0..2)
        # dmin: fp16 (2..4)
        # scales: 12 uint8 (4..16)
        # qh: 32 uint8 (16..48)
        # qs: 128 uint8 (48..176)
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        dmin = _fp16_bytes_to_f32_scalar(raw[2:4])
        scales = raw[4:16]
        qh = raw[16:48]
        qs = raw[48:176]

        sc = [0] * 8
        m = [0] * 8
        for j in range(4):
            sc[j] = scales[j] & 63
            m[j] = scales[j + 4] & 63
        for j in range(4, 8):
            sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
            m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)

        out = []
        for c in range(4):
            qs_chunk = qs[c * 32 : (c + 1) * 32]
            d1 = _f32_round(d * sc[2 * c])
            m1 = _f32_round(dmin * m[2 * c])
            d2 = _f32_round(d * sc[2 * c + 1])
            m2 = _f32_round(dmin * m[2 * c + 1])
            bit1 = 2 * c
            bit2 = 2 * c + 1
            for l in range(32):
                q_low = (qs_chunk[l] & 0x0F) | (((qh[l] >> bit1) & 1) << 4)
                out.append(_f32_round(_f32_round(d1 * q_low) - m1))
            for l in range(32):
                q_high = (qs_chunk[l] >> 4) | (((qh[l] >> bit2) & 1) << 4)
                out.append(_f32_round(_f32_round(d2 * q_high) - m2))
        return out

    if ggml_type == GGML_TYPE_Q6_K:
        # Layout (210 bytes):
        # ql: 128 uint8 (0..128)
        # qh: 64 uint8 (128..192)
        # scales: 16 int8 (192..208)
        # d: fp16 (208..210)
        ql = raw[0:128]
        qh = raw[128:192]
        scales_i8 = struct.unpack("<16b", raw[192:208])
        d = _fp16_bytes_to_f32_scalar(raw[208:210])

        out = [0.0] * 256
        for n in range(2):
            ql_half = ql[n * 64 : (n + 1) * 64]
            ql_0 = ql_half[0:32]
            ql_1 = ql_half[32:64]
            qh_half = qh[n * 32 : (n + 1) * 32]
            sc_half = scales_i8[n * 8 : (n + 1) * 8]
            for l in range(32):
                is_sub = l // 16
                q1 = ((ql_0[l] & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql_1[l] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql_0[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql_1[l] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) - 32
                out[n * 128 + l + 0] = _f32_round(_f32_round(d * sc_half[is_sub + 0]) * q1)
                out[n * 128 + l + 32] = _f32_round(_f32_round(d * sc_half[is_sub + 2]) * q2)
                out[n * 128 + l + 64] = _f32_round(_f32_round(d * sc_half[is_sub + 4]) * q3)
                out[n * 128 + l + 96] = _f32_round(_f32_round(d * sc_half[is_sub + 6]) * q4)
        return out

    if ggml_type == GGML_TYPE_IQ4_NL:
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        qs = raw[2:18]
        low = [_f32_round(d * KVALUES_IQ4NL[b & 0x0F]) for b in qs]
        high = [_f32_round(d * KVALUES_IQ4NL[(b >> 4) & 0x0F]) for b in qs]
        return low + high

    if ggml_type == GGML_TYPE_IQ3_S:
        # Layout (110 bytes):
        # d: fp16 (0..2)
        # qs: 64 uint8 (2..66)
        # qh: 8 uint8 (66..74)
        # signs: 32 uint8 (74..106)
        # scales: 4 uint8 (106..110)
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        qs = raw[2:66]
        qh = raw[66:74]
        signs = raw[74:106]
        scales = raw[106:110]
        out_iq3: List[float] = []
        qs_off = 0
        qh_off = 0
        signs_off = 0
        for ib32 in range(0, 8, 2):
            sc_byte = scales[ib32 // 2]
            db1 = _f32_round(d * (1 + 2 * (sc_byte & 0x0F)))
            db2 = _f32_round(d * (1 + 2 * (sc_byte >> 4)))
            for l in range(4):
                idx1 = qs[qs_off + 2 * l + 0] | ((qh[qh_off + 0] << (8 - 2 * l)) & 256)
                idx2 = qs[qs_off + 2 * l + 1] | ((qh[qh_off + 0] << (7 - 2 * l)) & 256)
                g1 = IQ3S_GRID_U32[idx1]
                g2 = IQ3S_GRID_U32[idx2]
                s_byte = signs[signs_off + l]
                for j in range(4):
                    v1 = (g1 >> (8 * j)) & 0xFF
                    sgn1 = -1.0 if (s_byte & (1 << j)) else 1.0
                    out_iq3.append(_f32_round(_f32_round(db1 * v1) * sgn1))
                for j in range(4):
                    v2 = (g2 >> (8 * j)) & 0xFF
                    sgn2 = -1.0 if (s_byte & (1 << (j + 4))) else 1.0
                    out_iq3.append(_f32_round(_f32_round(db1 * v2) * sgn2))
            qs_off += 8
            signs_off += 4
            for l in range(4):
                idx1 = qs[qs_off + 2 * l + 0] | ((qh[qh_off + 1] << (8 - 2 * l)) & 256)
                idx2 = qs[qs_off + 2 * l + 1] | ((qh[qh_off + 1] << (7 - 2 * l)) & 256)
                g1 = IQ3S_GRID_U32[idx1]
                g2 = IQ3S_GRID_U32[idx2]
                s_byte = signs[signs_off + l]
                for j in range(4):
                    v1 = (g1 >> (8 * j)) & 0xFF
                    sgn1 = -1.0 if (s_byte & (1 << j)) else 1.0
                    out_iq3.append(_f32_round(_f32_round(db2 * v1) * sgn1))
                for j in range(4):
                    v2 = (g2 >> (8 * j)) & 0xFF
                    sgn2 = -1.0 if (s_byte & (1 << (j + 4))) else 1.0
                    out_iq3.append(_f32_round(_f32_round(db2 * v2) * sgn2))
            qh_off += 2
            qs_off += 8
            signs_off += 4
        return out_iq3

    if ggml_type == GGML_TYPE_IQ4_XS:
        d = _fp16_bytes_to_f32_scalar(raw[0:2])
        scales_h = struct.unpack("<H", raw[2:4])[0]
        scales_l = raw[4:8]
        qs = raw[8:136]
        out: List[float] = []
        for ib in range(8):
            ls = ((scales_l[ib // 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4)
            dl = _f32_round(d * (ls - 32))
            qs_ib = qs[ib * 16 : (ib + 1) * 16]
            for j in range(16):
                out.append(_f32_round(dl * KVALUES_IQ4NL[qs_ib[j] & 0x0F]))
            for j in range(16):
                out.append(_f32_round(dl * KVALUES_IQ4NL[(qs_ib[j] >> 4) & 0x0F]))
        return out

    raise ValueError(f"Unsupported GGML type ID: {ggml_type}")


def dequantize_bytes(
    raw_bytes: Union[bytes, bytearray, memoryview],
    ggml_type: int,
    num_elements: int,
) -> np.ndarray:
    """Vectorized NumPy dequantization of GGML tensor buffers into 1D `np.float32` array.

    Parameters
    ----------
    raw_bytes : bytes | bytearray | memoryview
        Raw tensor bytes from GGUF file.
    ggml_type : int
        GGML type ID (`GGML_TYPE_F32`, `GGML_TYPE_Q4_K`, `GGML_TYPE_Q6_K`, etc.).
    num_elements : int
        Total number of logical elements to decode.

    Returns
    -------
    np.ndarray
        Contiguous 1D array of `np.float32` with shape `(num_elements,)`.
    """
    expected_bytes = compute_tensor_bytes(ggml_type, num_elements)
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Byte buffer length mismatch for type {GGML_TYPE_NAMES.get(ggml_type, str(ggml_type))}: "
            f"expected {expected_bytes} bytes for {num_elements} elements, got {len(raw_bytes)}"
        )

    if num_elements == 0:
        return np.empty(0, dtype=np.float32)

    if ggml_type == GGML_TYPE_F32:
        return np.frombuffer(raw_bytes, dtype="<f4").astype(np.float32, copy=True)

    if ggml_type == GGML_TYPE_F16:
        return np.frombuffer(raw_bytes, dtype="<f2").astype(np.float32)

    if ggml_type == GGML_TYPE_BF16:
        u16 = np.frombuffer(raw_bytes, dtype="<u2").astype(np.uint32)
        return (u16 << np.uint32(16)).view(np.float32)

    block_size, type_size = GGML_TYPE_INFO[ggml_type]
    n_blocks = num_elements // block_size
    buf = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, type_size)

    if ggml_type == GGML_TYPE_Q4_0:
        # 18 bytes: fp16 d (0..2), 16 uint8 qs (2..18)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        qs = buf[:, 2:18]  # (n_blocks, 16)
        low = ((qs & 0x0F).astype(np.int8) - 8).astype(np.float32) * d
        high = ((qs >> 4).astype(np.int8) - 8).astype(np.float32) * d
        out = np.concatenate([low, high], axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q4_1:
        # 20 bytes: fp16 d (0..2), fp16 m (2..4), 16 uint8 qs (4..20)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        m = buf[:, 2:4].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        qs = buf[:, 4:20]
        low = (qs & 0x0F).astype(np.float32) * d + m
        high = (qs >> 4).astype(np.float32) * d + m
        out = np.concatenate([low, high], axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q5_0:
        # 22 bytes: fp16 d (0..2), uint32 qh (2..6), 16 uint8 qs (6..22)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        qh = buf[:, 2:6].copy().view("<u4")  # (n_blocks, 1)
        qs = buf[:, 6:22]  # (n_blocks, 16)
        shifts = np.arange(16, dtype=np.uint32).reshape(1, 16)
        xh_0 = (((qh >> shifts) & 1) << 4).astype(np.uint8)
        xh_1 = (((qh >> (shifts + np.uint32(16))) & 1) << 4).astype(np.uint8)
        q0 = (((qs & 0x0F) | xh_0).astype(np.int8) - 16).astype(np.float32) * d
        q1 = (((qs >> 4) | xh_1).astype(np.int8) - 16).astype(np.float32) * d
        out = np.concatenate([q0, q1], axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q5_1:
        # 24 bytes: fp16 d (0..2), fp16 m (2..4), uint32 qh (4..8), 16 uint8 qs (8..24)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)
        m = buf[:, 2:4].copy().view("<f2").astype(np.float32)
        qh = buf[:, 4:8].copy().view("<u4")
        qs = buf[:, 8:24]
        shifts = np.arange(16, dtype=np.uint32).reshape(1, 16)
        xh_0 = (((qh >> shifts) & 1) << 4).astype(np.uint8)
        xh_1 = (((qh >> (shifts + np.uint32(16))) & 1) << 4).astype(np.uint8)
        q0 = ((qs & 0x0F) | xh_0).astype(np.float32) * d + m
        q1 = ((qs >> 4) | xh_1).astype(np.float32) * d + m
        out = np.concatenate([q0, q1], axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q8_0:
        # 34 bytes: fp16 d (0..2), 32 int8 qs (2..34)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        qs = buf[:, 2:34].view(np.int8).astype(np.float32)  # (n_blocks, 32)
        out = d * qs
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q2_K:
        # 84 bytes: scales (0..16), qs (16..80), fp16 d (80..82), fp16 dmin (82..84)
        scales = buf[:, 0:16]  # (n_blocks, 16)
        qs = buf[:, 16:80].reshape(n_blocks, 2, 32)  # (n_blocks, 2, 32)
        d = buf[:, 80:82].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        dmin = buf[:, 82:84].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)

        dl = d * (scales & 0x0F).astype(np.float32)  # (n_blocks, 16)
        ml = dmin * (scales >> 4).astype(np.float32)  # (n_blocks, 16)

        chunks = []
        is_idx = 0
        for n in range(2):
            q_chunk = qs[:, n, :]  # (n_blocks, 32)
            for shift in (0, 2, 4, 6):
                q0 = ((q_chunk[:, 0:16] >> shift) & 3).astype(np.float32)
                q1 = ((q_chunk[:, 16:32] >> shift) & 3).astype(np.float32)
                chunks.append(dl[:, is_idx : is_idx + 1] * q0 - ml[:, is_idx : is_idx + 1])
                chunks.append(dl[:, is_idx + 1 : is_idx + 2] * q1 - ml[:, is_idx + 1 : is_idx + 2])
                is_idx += 2
        out = np.concatenate(chunks, axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q3_K:
        # 110 bytes: hmask (0..32), qs (32..96), scales (96..108), fp16 d (108..110)
        hmask = buf[:, 0:32]  # (n_blocks, 32)
        qs = buf[:, 32:96].reshape(n_blocks, 2, 32)
        sc_raw = buf[:, 96:108]  # (n_blocks, 12)
        d = buf[:, 108:110].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)

        sc = np.zeros((n_blocks, 16), dtype=np.int16)
        sc[:, 0:8] = sc_raw[:, 0:8] & 0x0F
        sc[:, 8:16] = sc_raw[:, 0:8] >> 4
        for i in range(4):
            b = sc_raw[:, 8 + i]
            sc[:, i] |= ((b & 0x03) << 4).astype(np.int16)
            sc[:, i + 4] |= (((b >> 2) & 0x03) << 4).astype(np.int16)
            sc[:, i + 8] |= (((b >> 4) & 0x03) << 4).astype(np.int16)
            sc[:, i + 12] |= (((b >> 6) & 0x03) << 4).astype(np.int16)
        dl = d * (sc - 32).astype(np.float32)  # (n_blocks, 16)

        chunks = []
        is_idx = 0
        m_bit = 1
        for n in range(2):
            q_chunk = qs[:, n, :]
            for shift in (0, 2, 4, 6):
                low0 = ((q_chunk[:, 0:16] >> shift) & 3).astype(np.int8)
                hi0 = np.where((hmask[:, 0:16] & m_bit) != 0, 0, 4).astype(np.int8)
                q0 = (low0 - hi0).astype(np.float32)
                chunks.append(dl[:, is_idx : is_idx + 1] * q0)
                is_idx += 1

                low1 = ((q_chunk[:, 16:32] >> shift) & 3).astype(np.int8)
                hi1 = np.where((hmask[:, 16:32] & m_bit) != 0, 0, 4).astype(np.int8)
                q1 = (low1 - hi1).astype(np.float32)
                chunks.append(dl[:, is_idx : is_idx + 1] * q1)
                is_idx += 1
                m_bit <<= 1
        out = np.concatenate(chunks, axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q4_K:
        # 144 bytes:
        # d: fp16 (0..2)
        # dmin: fp16 (2..4)
        # scales: 12 uint8 (4..16)
        # qs: 128 uint8 (16..144)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        dmin = buf[:, 2:4].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        scales = buf[:, 4:16]  # (n_blocks, 12)
        qs = buf[:, 16:144].reshape(n_blocks, 4, 32)  # (n_blocks, 4, 32)

        sc = np.empty((n_blocks, 8), dtype=np.uint8)
        m = np.empty((n_blocks, 8), dtype=np.uint8)
        sc[:, 0:4] = scales[:, 0:4] & 63
        m[:, 0:4] = scales[:, 4:8] & 63
        sc[:, 4:8] = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
        m[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)

        d_sc = d * sc.astype(np.float32)  # (n_blocks, 8)
        dm_m = dmin * m.astype(np.float32)  # (n_blocks, 8)

        d1 = d_sc[:, 0::2].reshape(n_blocks, 4, 1)  # (n_blocks, 4, 1)
        m1 = dm_m[:, 0::2].reshape(n_blocks, 4, 1)
        d2 = d_sc[:, 1::2].reshape(n_blocks, 4, 1)
        m2 = dm_m[:, 1::2].reshape(n_blocks, 4, 1)

        q_low = (qs & 0x0F).astype(np.float32)  # (n_blocks, 4, 32)
        q_high = (qs >> 4).astype(np.float32)  # (n_blocks, 4, 32)

        out_low = d1 * q_low - m1  # (n_blocks, 4, 32)
        out_high = d2 * q_high - m2  # (n_blocks, 4, 32)
        out = np.concatenate([out_low, out_high], axis=2)  # (n_blocks, 4, 64)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q5_K:
        # 176 bytes:
        # d: fp16 (0..2), dmin: fp16 (2..4), scales: 12 uint8 (4..16), qh: 32 uint8 (16..48), qs: 128 uint8 (48..176)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)
        dmin = buf[:, 2:4].copy().view("<f2").astype(np.float32)
        scales = buf[:, 4:16]
        qh = buf[:, 16:48]  # (n_blocks, 32)
        qs = buf[:, 48:176].reshape(n_blocks, 4, 32)

        sc = np.empty((n_blocks, 8), dtype=np.uint8)
        m = np.empty((n_blocks, 8), dtype=np.uint8)
        sc[:, 0:4] = scales[:, 0:4] & 63
        m[:, 0:4] = scales[:, 4:8] & 63
        sc[:, 4:8] = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
        m[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)

        d_sc = d * sc.astype(np.float32)
        dm_m = dmin * m.astype(np.float32)

        chunks = []
        for c in range(4):
            qs_c = qs[:, c, :]  # (n_blocks, 32)
            bit1 = np.uint8(2 * c)
            bit2 = np.uint8(2 * c + 1)
            q_low = ((qs_c & 0x0F) | (((qh >> bit1) & 1) << 4)).astype(np.float32)
            q_high = ((qs_c >> 4) | (((qh >> bit2) & 1) << 4)).astype(np.float32)
            d1 = d_sc[:, 2 * c : 2 * c + 1]
            m1 = dm_m[:, 2 * c : 2 * c + 1]
            d2 = d_sc[:, 2 * c + 1 : 2 * c + 2]
            m2 = dm_m[:, 2 * c + 1 : 2 * c + 2]
            chunks.append(d1 * q_low - m1)
            chunks.append(d2 * q_high - m2)
        out = np.concatenate(chunks, axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_Q6_K:
        # 210 bytes:
        # ql: 128 uint8 (0..128) -> (n_blocks, 2, 64)
        # qh: 64 uint8 (128..192) -> (n_blocks, 2, 32)
        # scales: 16 int8 (192..208) -> (n_blocks, 2, 8)
        # d: fp16 (208..210) -> (n_blocks, 1, 1)
        ql = buf[:, 0:128].reshape(n_blocks, 2, 64)
        qh = buf[:, 128:192].reshape(n_blocks, 2, 32)
        scales = buf[:, 192:208].view(np.int8).reshape(n_blocks, 2, 8).astype(np.float32)
        d = buf[:, 208:210].copy().view("<f2").astype(np.float32).reshape(n_blocks, 1, 1)

        ql_0 = ql[:, :, 0:32]  # (n_blocks, 2, 32)
        ql_1 = ql[:, :, 32:64]  # (n_blocks, 2, 32)

        q1 = (((ql_0 & 0x0F) | (((qh >> 0) & 3) << 4)).astype(np.int8) - 32).astype(np.float32)
        q2 = (((ql_1 & 0x0F) | (((qh >> 2) & 3) << 4)).astype(np.int8) - 32).astype(np.float32)
        q3 = (((ql_0 >> 4) | (((qh >> 4) & 3) << 4)).astype(np.int8) - 32).astype(np.float32)
        q4 = (((ql_1 >> 4) | (((qh >> 6) & 3) << 4)).astype(np.int8) - 32).astype(np.float32)

        # Reshape each (n_blocks, 2, 32) into (n_blocks, 2, 2, 16) so scale pair [0, 1] broadcasts over 16 elements
        d_sc = d * scales  # (n_blocks, 2, 8)
        sc1 = d_sc[:, :, 0:2].reshape(n_blocks, 2, 2, 1)
        sc2 = d_sc[:, :, 2:4].reshape(n_blocks, 2, 2, 1)
        sc3 = d_sc[:, :, 4:6].reshape(n_blocks, 2, 2, 1)
        sc4 = d_sc[:, :, 6:8].reshape(n_blocks, 2, 2, 1)

        o1 = (sc1 * q1.reshape(n_blocks, 2, 2, 16)).reshape(n_blocks, 2, 32)
        o2 = (sc2 * q2.reshape(n_blocks, 2, 2, 16)).reshape(n_blocks, 2, 32)
        o3 = (sc3 * q3.reshape(n_blocks, 2, 2, 16)).reshape(n_blocks, 2, 32)
        o4 = (sc4 * q4.reshape(n_blocks, 2, 2, 16)).reshape(n_blocks, 2, 32)

        out = np.concatenate([o1, o2, o3, o4], axis=2)  # (n_blocks, 2, 128)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_IQ4_NL:
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        qs = buf[:, 2:18]  # (n_blocks, 16)
        low = d * _KVALUES_IQ4NL_NP[qs & 0x0F]
        high = d * _KVALUES_IQ4NL_NP[(qs >> 4) & 0x0F]
        out = np.concatenate([low, high], axis=1)
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_IQ3_S:
        # 110 bytes: fp16 d (0..2), qs (2..66), qh (66..74), signs (74..106), scales (106..110)
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        qs = buf[:, 2:66].reshape(n_blocks, 4, 2, 8).astype(np.uint16)
        qh = buf[:, 66:74].reshape(n_blocks, 4, 2).astype(np.uint16)
        signs = buf[:, 74:106].reshape(n_blocks, 4, 2, 4)
        scales = buf[:, 106:110]  # (n_blocks, 4)

        sc_low = (1 + 2 * (scales & 0x0F)).astype(np.float32)   # (n_blocks, 4)
        sc_high = (1 + 2 * (scales >> 4)).astype(np.float32)    # (n_blocks, 4)
        dbs = np.stack([d * sc_low, d * sc_high], axis=2)       # (n_blocks, 4, 2)

        l_arr = np.arange(4, dtype=np.uint16)
        idx1 = qs[:, :, :, 0::2] | ((qh[:, :, :, None] << (8 - 2 * l_arr)) & 256)  # (n_blocks, 4, 2, 4)
        idx2 = qs[:, :, :, 1::2] | ((qh[:, :, :, None] << (7 - 2 * l_arr)) & 256)  # (n_blocks, 4, 2, 4)

        g1 = _IQ3S_GRID_NP[idx1]  # (n_blocks, 4, 2, 4, 4)
        g2 = _IQ3S_GRID_NP[idx2]  # (n_blocks, 4, 2, 4, 4)
        grid_8 = np.concatenate([g1, g2], axis=-1)  # (n_blocks, 4, 2, 4, 8)

        bit_shifts = np.arange(8, dtype=np.uint8)
        sign_bits = (signs[:, :, :, :, None] >> bit_shifts) & 1
        sign_mult = np.where(sign_bits != 0, np.float32(-1.0), np.float32(1.0))  # (n_blocks, 4, 2, 4, 8)

        out = dbs[:, :, :, None, None] * grid_8 * sign_mult
        return out.reshape(num_elements)

    if ggml_type == GGML_TYPE_IQ4_XS:
        d = buf[:, 0:2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
        scales_h = buf[:, 2:4].copy().view("<u2").reshape(n_blocks)  # (n_blocks,)
        scales_l = buf[:, 4:8]  # (n_blocks, 4)
        qs = buf[:, 8:136].reshape(n_blocks, 8, 16)
        chunks = []
        for ib in range(8):
            sl = (scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0x0F
            sh = ((scales_h >> (2 * ib)) & 3).astype(np.uint8) << 4
            ls = (sl | sh).astype(np.float32) - 32.0
            dl = d * ls[:, None]  # (n_blocks, 1)
            qs_ib = qs[:, ib, :]
            chunks.append(dl * _KVALUES_IQ4NL_NP[qs_ib & 0x0F])
            chunks.append(dl * _KVALUES_IQ4NL_NP[(qs_ib >> 4) & 0x0F])
        out = np.concatenate(chunks, axis=1)
        return out.reshape(num_elements)

    raise ValueError(f"Unsupported GGML type ID: {ggml_type}")
