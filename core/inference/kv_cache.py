"""
Native NumPy KV Cache with upfront memory budget validation for SeedOssForCausalLM.
Written from scratch using only the Python standard library and NumPy.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple

import numpy as np


class InsufficientMemoryError(MemoryError):
    """Raised when the requested KV cache exceeds available system or configured memory budget."""


class ContextLengthExceededError(ValueError):
    """Raised when sequence position or prompt length exceeds the configured max_seq_len."""


def get_available_memory_bytes() -> int:
    """
    Query available system RAM in bytes using only Python standard library facilities.
    """
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys)
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except Exception:
            pass

        try:
            if hasattr(os, "sysconf"):
                page_size = os.sysconf("SC_PAGE_SIZE")
                avail_pages = os.sysconf("SC_AVPHYS_PAGES")
                if page_size > 0 and avail_pages > 0:
                    return int(page_size * avail_pages)
        except Exception:
            pass

    # Fallback conservative 8 GiB estimate if OS query is unavailable
    return 8 * 1024 * 1024 * 1024


class KVCache:
    """
    Grouped-Query Attention (GQA) Key-Value Cache for SeedOssForCausalLM.

    Stores K and V tensors of shape (num_kv_heads, cached_len, head_dim) per layer.
    Validates the worst-case memory footprint for max_seq_len upfront against
    available system RAM (or an explicit memory_limit_bytes) and raises
    InsufficientMemoryError cleanly rather than crashing or swapping.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: np.dtype | type = np.float32,
        memory_limit_bytes: Optional[int] = None,
        initial_capacity: int = 128,
    ) -> None:
        if num_layers <= 0 or num_kv_heads <= 0 or head_dim <= 0 or max_seq_len <= 0:
            raise ValueError(
                f"Invalid KVCache dimensions: num_layers={num_layers}, "
                f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, max_seq_len={max_seq_len}"
            )

        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.dtype = np.dtype(dtype)

        self.required_bytes = (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.max_seq_len
            * int(self.dtype.itemsize)
        )

        available_bytes = (
            int(memory_limit_bytes)
            if memory_limit_bytes is not None
            else get_available_memory_bytes()
        )
        self.available_bytes = available_bytes

        if self.required_bytes > available_bytes:
            req_gb = self.required_bytes / (1024 ** 3)
            avail_gb = available_bytes / (1024 ** 3)
            raise InsufficientMemoryError(
                f"KV cache requires {self.required_bytes} bytes ({req_gb:.2f} GiB) for "
                f"num_layers={self.num_layers}, num_kv_heads={self.num_kv_heads}, "
                f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
                f"which exceeds available memory budget of {available_bytes} bytes ({avail_gb:.2f} GiB)."
            )

        self._initial_capacity = min(max(16, int(initial_capacity)), self.max_seq_len)
        self._capacity = self._initial_capacity
        self._layer_Lengths = [0] * self.num_layers

        self._k_cache = [
            np.zeros((self.num_kv_heads, self._capacity, self.head_dim), dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        self._v_cache = [
            np.zeros((self.num_kv_heads, self._capacity, self.head_dim), dtype=self.dtype)
            for _ in range(self.num_layers)
        ]

    @property
    def seq_len(self) -> int:
        """Current cached sequence length (from layer 0, or max across layers)."""
        return max(self._layer_Lengths) if self._layer_Lengths else 0

    def _ensure_capacity(self, required_len: int) -> None:
        if required_len > self.max_seq_len:
            raise ContextLengthExceededError(
                f"Requested sequence length {required_len} exceeds max_seq_len={self.max_seq_len}."
            )
        if required_len <= self._capacity:
            return

        new_cap = self._capacity
        while new_cap < required_len:
            new_cap = min(new_cap * 2, self.max_seq_len)

        for layer_idx in range(self.num_layers):
            cur_len = self._layer_Lengths[layer_idx]
            new_k = np.zeros((self.num_kv_heads, new_cap, self.head_dim), dtype=self.dtype)
            new_v = np.zeros((self.num_kv_heads, new_cap, self.head_dim), dtype=self.dtype)
            if cur_len > 0:
                new_k[:, :cur_len, :] = self._k_cache[layer_idx][:, :cur_len, :]
                new_v[:, :cur_len, :] = self._v_cache[layer_idx][:, :cur_len, :]
            self._k_cache[layer_idx] = new_k
            self._v_cache[layer_idx] = new_v

        self._capacity = new_cap

    def update(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
        start_pos: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert new key/value states into the cache for layer `layer_idx` starting at `start_pos`
        and return the full valid (K, V) slices up to `start_pos + new_seq_len`.

        Expected k_new, v_new shape: (num_kv_heads, new_seq_len, head_dim).
        """
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        if k_new.ndim != 3 or v_new.ndim != 3:
            raise ValueError(
                f"Expected 3D tensors (num_kv_heads, seq_len, head_dim), got {k_new.shape} and {v_new.shape}"
            )
        if k_new.shape[0] != self.num_kv_heads or k_new.shape[2] != self.head_dim:
            raise ValueError(
                f"Expected shape ({self.num_kv_heads}, seq_len, {self.head_dim}), got {k_new.shape}"
            )
        if k_new.shape != v_new.shape:
            raise ValueError(f"k_new shape {k_new.shape} does not match v_new shape {v_new.shape}")

        pos = self._layer_Lengths[layer_idx] if start_pos is None else int(start_pos)
        if pos < 0:
            raise ValueError(f"start_pos must be non-negative, got {pos}")

        new_tokens = int(k_new.shape[1])
        end_pos = pos + new_tokens
        self._ensure_capacity(end_pos)

        self._k_cache[layer_idx][:, pos:end_pos, :] = k_new.astype(self.dtype, copy=False)
        self._v_cache[layer_idx][:, pos:end_pos, :] = v_new.astype(self.dtype, copy=False)
        self._layer_Lengths[layer_idx] = end_pos

        return (
            self._k_cache[layer_idx][:, :end_pos, :],
            self._v_cache[layer_idx][:, :end_pos, :],
        )

    def get(
        self,
        layer_idx: int,
        end_pos: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached (K, V) tensors for `layer_idx` up to `end_pos`."""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        limit = self._layer_Lengths[layer_idx] if end_pos is None else int(end_pos)
        if limit < 0 or limit > self._layer_Lengths[layer_idx]:
            raise ValueError(
                f"Invalid end_pos={limit} for layer {layer_idx} with cached length {self._layer_Lengths[layer_idx]}"
            )
        return (
            self._k_cache[layer_idx][:, :limit, :],
            self._v_cache[layer_idx][:, :limit, :],
        )

    def truncate(self, new_len: int) -> None:
        """Truncate cached sequence length across all layers to `new_len`."""
        if new_len < 0 or new_len > self.seq_len:
            raise ValueError(f"Cannot truncate KV cache of length {self.seq_len} to {new_len}")
        for i in range(self.num_layers):
            prev_len = self._layer_Lengths[i]
            if new_len < prev_len:
                self._k_cache[i][:, new_len:prev_len, :] = 0.0
                self._v_cache[i][:, new_len:prev_len, :] = 0.0
                self._layer_Lengths[i] = new_len

    def reset(self) -> None:
        """Reset all cached positions to zero."""
        for i in range(self.num_layers):
            cur_len = self._layer_Lengths[i]
            if cur_len > 0:
                self._k_cache[i][:, :cur_len, :] = 0.0
                self._v_cache[i][:, :cur_len, :] = 0.0
            self._layer_Lengths[i] = 0
