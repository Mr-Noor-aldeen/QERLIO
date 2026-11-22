"""From-scratch GGUF v2/v3 parser and zero-copy memory-mapped tensor loader.

Uses only Python stdlib (`struct`, `mmap`, `os`, `io`, `math`, `pathlib`, `dataclasses`, `typing`)
and `numpy` via `core.inference.dequant`.
"""

from __future__ import annotations

import io
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.inference.dequant import (
    GGML_TYPE_NAMES,
    compute_tensor_bytes,
    dequantize_bytes,
    get_block_size,
    get_type_size,
)

GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSIONS = (2, 3)
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value type IDs
GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12

# Safety limits against corrupt/malicious headers
_MAX_TENSOR_COUNT = 1_000_000
_MAX_KV_COUNT = 100_000
_MAX_STRING_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_ELEMENTS = 10_000_000
_MAX_TENSOR_DIMS = 8

_SCALAR_UNPACK_FORMATS: Dict[int, Tuple[str, int]] = {
    GGUF_VALUE_TYPE_UINT8: ("<B", 1),
    GGUF_VALUE_TYPE_INT8: ("<b", 1),
    GGUF_VALUE_TYPE_UINT16: ("<H", 2),
    GGUF_VALUE_TYPE_INT16: ("<h", 2),
    GGUF_VALUE_TYPE_UINT32: ("<I", 4),
    GGUF_VALUE_TYPE_INT32: ("<i", 4),
    GGUF_VALUE_TYPE_FLOAT32: ("<f", 4),
    GGUF_VALUE_TYPE_UINT64: ("<Q", 8),
    GGUF_VALUE_TYPE_INT64: ("<q", 8),
    GGUF_VALUE_TYPE_FLOAT64: ("<d", 8),
}


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Descriptor for a single tensor in a GGUF v2/v3 file."""

    name: str
    n_dims: int
    dims: Tuple[int, ...]
    type_id: int
    type_name: str
    offset: int
    num_elements: int
    n_bytes: int

    @property
    def numpy_shape(self) -> Tuple[int, ...]:
        """Return NumPy row-major shape `(ne_{n-1}, ..., ne_1, ne_0)` from GGUF `[ne_0, ne_1, ...]`."""
        return tuple(reversed(self.dims))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tensor descriptor matching GGUF metadata fixture schema."""
        return {
            "name": self.name,
            "n_dims": self.n_dims,
            "dims": list(self.dims),
            "type_id": self.type_id,
            "type": self.type_name,
            "offset": self.offset,
        }


class GGUFLoader:
    """Pure-stdlib + NumPy GGUF v2/v3 parser and zero-copy memory-mapped tensor loader."""

    def __init__(self, path: Union[str, Path], header_only: bool = False) -> None:
        self.path = Path(path)
        self.header_only = bool(header_only)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"GGUF path is not a regular file: {self.path}")

        self.file_size: int = self.path.stat().st_size
        if self.file_size < 24:
            raise ValueError(f"File too small to contain GGUF header ({self.file_size} bytes): {self.path}")

        self.version: int = 0
        self.tensor_count: int = 0
        self.kv_count: int = 0
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, GGUFTensorInfo] = {}
        self.tensor_list: List[GGUFTensorInfo] = []
        self.alignment: int = GGUF_DEFAULT_ALIGNMENT
        self.header_end_offset: int = 0
        self.data_offset: int = 0

        self._file_obj: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None

        self._parse_file()

    def _read_exact(self, f: BinaryIO, n: int, context: str) -> bytes:
        data = f.read(n)
        if len(data) != n:
            raise ValueError(
                f"Unexpected EOF while reading {context}: expected {n} bytes, got {len(data)}"
            )
        return data

    def _read_string(self, f: BinaryIO, context: str) -> str:
        raw_len = self._read_exact(f, 8, f"{context} length")
        length = struct.unpack("<Q", raw_len)[0]
        if length > _MAX_STRING_BYTES or f.tell() + length > self.file_size:
            raise ValueError(
                f"Corrupt or excessive string length {length} while reading {context}"
            )
        raw_bytes = self._read_exact(f, length, f"{context} payload")
        return raw_bytes.decode("utf-8", errors="replace")

    def _read_value(self, f: BinaryIO, vtype: int, context: str) -> Any:
        if vtype in _SCALAR_UNPACK_FORMATS:
            fmt, size = _SCALAR_UNPACK_FORMATS[vtype]
            raw = self._read_exact(f, size, context)
            return struct.unpack(fmt, raw)[0]

        if vtype == GGUF_VALUE_TYPE_BOOL:
            raw = self._read_exact(f, 1, context)
            val = struct.unpack("<B", raw)[0]
            if val not in (0, 1):
                raise ValueError(f"Invalid GGUF boolean byte {val} at {context}")
            return bool(val)

        if vtype == GGUF_VALUE_TYPE_STRING:
            return self._read_string(f, context)

        if vtype == GGUF_VALUE_TYPE_ARRAY:
            hdr = self._read_exact(f, 12, f"{context} array header")
            elem_type, count = struct.unpack("<IQ", hdr)
            if count > _MAX_ARRAY_ELEMENTS:
                raise ValueError(
                    f"Corrupt or excessive GGUF array length {count} at {context}"
                )
            if elem_type == GGUF_VALUE_TYPE_ARRAY:
                return [self._read_value(f, elem_type, f"{context}[{i}]") for i in range(count)]
            if elem_type in _SCALAR_UNPACK_FORMATS:
                fmt, elem_size = _SCALAR_UNPACK_FORMATS[elem_type]
                total_bytes = count * elem_size
                if f.tell() + total_bytes > self.file_size:
                    raise ValueError(
                        f"Truncated GGUF scalar array at {context}: requires {total_bytes} bytes"
                    )
                raw_arr = self._read_exact(f, total_bytes, f"{context} array data")
                if count == 0:
                    return []
                return list(struct.unpack(f"<{count}{fmt[1]}", raw_arr))
            return [self._read_value(f, elem_type, f"{context}[{i}]") for i in range(count)]

        raise ValueError(f"Unsupported or corrupt GGUF value type ID {vtype} at {context}")

    def _parse_file(self) -> None:
        with open(self.path, "rb") as f:
            magic = self._read_exact(f, 4, "GGUF magic")
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic bytes: expected {GGUF_MAGIC!r}, got {magic!r}")

            hdr = self._read_exact(f, 20, "GGUF header counts")
            version, tensor_count, kv_count = struct.unpack("<IQQ", hdr)
            if version not in GGUF_SUPPORTED_VERSIONS:
                raise ValueError(
                    f"Unsupported GGUF version {version}; supported versions: {GGUF_SUPPORTED_VERSIONS}"
                )
            if tensor_count > _MAX_TENSOR_COUNT:
                raise ValueError(f"Corrupt or excessive tensor_count: {tensor_count}")
            if kv_count > _MAX_KV_COUNT:
                raise ValueError(f"Corrupt or excessive kv_count: {kv_count}")

            self.version = version
            self.tensor_count = tensor_count
            self.kv_count = kv_count

            for i in range(kv_count):
                key = self._read_string(f, f"KV[{i}] key")
                if not key:
                    raise ValueError(f"Empty metadata key at KV[{i}]")
                vtype_raw = self._read_exact(f, 4, f"KV[{i}] ({key}) type")
                vtype = struct.unpack("<I", vtype_raw)[0]
                val = self._read_value(f, vtype, f"KV[{i}] ({key})")
                self.metadata[key] = val

            raw_align = self.metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
            if not isinstance(raw_align, int) or raw_align <= 0 or (raw_align & (raw_align - 1)) != 0:
                raise ValueError(f"Invalid general.alignment value: {raw_align!r}")
            self.alignment = int(raw_align)

            for i in range(tensor_count):
                tname = self._read_string(f, f"tensor[{i}] name")
                if not tname:
                    raise ValueError(f"Empty tensor name at index {i}")
                if tname in self.tensors:
                    raise ValueError(f"Duplicate tensor name in GGUF header: {tname!r}")

                ndims_raw = self._read_exact(f, 4, f"tensor[{i}] ({tname}) n_dims")
                n_dims = struct.unpack("<I", ndims_raw)[0]
                if n_dims == 0 or n_dims > _MAX_TENSOR_DIMS:
                    raise ValueError(
                        f"Invalid tensor dimension count {n_dims} for tensor {tname!r}"
                    )

                dims_raw = self._read_exact(f, 8 * n_dims, f"tensor[{i}] ({tname}) dims")
                dims = struct.unpack(f"<{n_dims}Q", dims_raw)
                if any(d <= 0 for d in dims):
                    raise ValueError(f"Invalid non-positive dimension in {dims} for tensor {tname!r}")

                type_off_raw = self._read_exact(f, 12, f"tensor[{i}] ({tname}) type & offset")
                type_id, offset = struct.unpack("<IQ", type_off_raw)
                if type_id not in GGML_TYPE_NAMES:
                    raise ValueError(
                        f"Unsupported GGML tensor type ID {type_id} for tensor {tname!r}"
                    )

                num_elements = math.prod(dims)
                n_bytes = compute_tensor_bytes(type_id, num_elements)
                tinfo = GGUFTensorInfo(
                    name=tname,
                    n_dims=n_dims,
                    dims=tuple(int(d) for d in dims),
                    type_id=int(type_id),
                    type_name=GGML_TYPE_NAMES[type_id],
                    offset=int(offset),
                    num_elements=int(num_elements),
                    n_bytes=int(n_bytes),
                )
                self.tensors[tname] = tinfo
                self.tensor_list.append(tinfo)

            self.header_end_offset = f.tell()
            self.data_offset = (
                (self.header_end_offset + self.alignment - 1) // self.alignment
            ) * self.alignment

        if not self.header_only:
            if self.data_offset > self.file_size and self.tensor_count > 0:
                raise ValueError(
                    f"Truncated GGUF file: data_offset ({self.data_offset}) exceeds "
                    f"file_size ({self.file_size})"
                )
            for tinfo in self.tensor_list:
                end_pos = self.data_offset + tinfo.offset + tinfo.n_bytes
                if end_pos > self.file_size:
                    raise ValueError(
                        f"Truncated GGUF tensor data for {tinfo.name!r}: requires bytes "
                        f"[{self.data_offset + tinfo.offset}..{end_pos}], but file size is {self.file_size}"
                    )
            self._file_obj = open(self.path, "rb")
            self._mmap = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)

    def _ensure_mmap(self) -> mmap.mmap:
        if self._mmap is None:
            if self.header_only:
                self._file_obj = open(self.path, "rb")
                self._mmap = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                raise RuntimeError("GGUFLoader memory map is already closed")
        return self._mmap

    def get_tensor_info(self, name: str) -> GGUFTensorInfo:
        """Return the `GGUFTensorInfo` descriptor for `name`."""
        if name not in self.tensors:
            raise KeyError(f"Tensor {name!r} not found in GGUF file")
        return self.tensors[name]

    def get_tensor_raw(self, name: str) -> memoryview:
        """Return a zero-copy `memoryview` slice of the raw quantized bytes for tensor `name`."""
        tinfo = self.get_tensor_info(name)
        mm = self._ensure_mmap()
        start = self.data_offset + tinfo.offset
        end = start + tinfo.n_bytes
        if end > self.file_size:
            raise ValueError(
                f"Tensor {name!r} range [{start}..{end}] exceeds file size {self.file_size}"
            )
        return memoryview(mm)[start:end]

    def get_tensor_f32(self, name: str) -> np.ndarray:
        """Dequantize tensor `name` into an `np.float32` array reshaped to `numpy_shape`."""
        tinfo = self.get_tensor_info(name)
        raw_view = self.get_tensor_raw(name)
        try:
            flat = dequantize_bytes(raw_view, tinfo.type_id, tinfo.num_elements)
        finally:
            raw_view.release()
        return flat.reshape(tinfo.numpy_shape)

    def get_tensor_rows_f32(
        self,
        name: str,
        row_indices: Union[Sequence[int], np.ndarray],
    ) -> np.ndarray:
        """Dequantize only specific outer rows of a 2D tensor without dequantizing the full matrix.

        Useful for fast token embedding lookups (`token_embd.weight[token_ids]`) and chunked
        matrix multiplication on large quantized tensors.
        """
        tinfo = self.get_tensor_info(name)
        if tinfo.n_dims != 2:
            raise ValueError(
                f"get_tensor_rows_f32 requires a 2D tensor, got {tinfo.n_dims}D for {name!r}"
            )
        n_cols, n_rows = tinfo.dims[0], tinfo.dims[1]
        block_size = get_block_size(tinfo.type_id)
        type_size = get_type_size(tinfo.type_id)
        if n_cols % block_size != 0:
            raise ValueError(
                f"Row width {n_cols} is not divisible by block size {block_size} for {name!r}"
            )
        row_bytes = (n_cols // block_size) * type_size
        mm = self._ensure_mmap()
        base_offset = self.data_offset + tinfo.offset

        indices = [int(idx) for idx in row_indices]
        out = np.empty((len(indices), n_cols), dtype=np.float32)
        mv = memoryview(mm)
        try:
            for out_i, row_idx in enumerate(indices):
                if row_idx < 0 or row_idx >= n_rows:
                    raise IndexError(
                        f"Row index {row_idx} out of bounds for tensor {name!r} with {n_rows} rows"
                    )
                start = base_offset + row_idx * row_bytes
                end = start + row_bytes
                if end > self.file_size:
                    raise ValueError(
                        f"Row {row_idx} of tensor {name!r} [{start}..{end}] exceeds file size {self.file_size}"
                    )
                row_slice = mv[start:end]
                try:
                    out[out_i] = dequantize_bytes(row_slice, tinfo.type_id, n_cols)
                finally:
                    row_slice.release()
        finally:
            mv.release()
        return out

    def extract_architecture_summary(self) -> Dict[str, Any]:
        """Extract standardized architecture & tokenizer summary from GGUF metadata."""
        arch = str(self.metadata.get("general.architecture", ""))
        prefix = f"{arch}." if arch else ""
        tokens = self.metadata.get("tokenizer.ggml.tokens")
        merges = self.metadata.get("tokenizer.ggml.merges")
        vocab_size = len(tokens) if isinstance(tokens, list) else 0
        merges_count = len(merges) if isinstance(merges, list) else 0

        return {
            "magic": "GGUF",
            "version": self.version,
            "tensor_count": self.tensor_count,
            "kv_count": self.kv_count,
            "alignment": self.alignment,
            "header_end_offset": self.header_end_offset,
            "data_offset": self.data_offset,
            "architecture": arch,
            "name": self.metadata.get("general.name"),
            "finetune": self.metadata.get("general.finetune"),
            "basename": self.metadata.get("general.basename"),
            "size_label": self.metadata.get("general.size_label"),
            "file_type": self.metadata.get("general.file_type"),
            "quantization_version": self.metadata.get("general.quantization_version"),
            "block_count": self.metadata.get(f"{prefix}block_count"),
            "context_length": self.metadata.get(f"{prefix}context_length"),
            "embedding_length": self.metadata.get(f"{prefix}embedding_length"),
            "feed_forward_length": self.metadata.get(f"{prefix}feed_forward_length"),
            "attention.head_count": self.metadata.get(f"{prefix}attention.head_count"),
            "attention.head_count_kv": self.metadata.get(f"{prefix}attention.head_count_kv"),
            "attention.key_length": self.metadata.get(f"{prefix}attention.key_length"),
            "attention.value_length": self.metadata.get(f"{prefix}attention.value_length"),
            "attention.layer_norm_rms_epsilon": self.metadata.get(
                f"{prefix}attention.layer_norm_rms_epsilon"
            ),
            "rope.freq_base": self.metadata.get(f"{prefix}rope.freq_base"),
            "tokenizer.model": self.metadata.get("tokenizer.ggml.model"),
            "tokenizer.pre": self.metadata.get("tokenizer.ggml.pre"),
            "bos_token_id": self.metadata.get("tokenizer.ggml.bos_token_id"),
            "eos_token_id": self.metadata.get("tokenizer.ggml.eos_token_id"),
            "padding_token_id": self.metadata.get("tokenizer.ggml.padding_token_id"),
            "vocab_size": vocab_size,
            "merges_count": merges_count,
        }

    def close(self) -> None:
        """Close open memory map and underlying file handle."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file_obj is not None:
            self._file_obj.close()
            self._file_obj = None

    def __enter__(self) -> "GGUFLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
