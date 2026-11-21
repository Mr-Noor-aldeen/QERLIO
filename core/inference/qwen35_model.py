"""From-scratch NumPy implementation of Qwen3.5 / Qwen3.8 (`qwen35`) Hybrid Architecture.

Implements the hybrid Gated DeltaNet (Linear Attention / SSM) + Gated Full-Attention
architecture (`general.architecture = "qwen35"`) directly from GGUF v2/v3 weights
using ONLY the Python standard library and NumPy.

Architecture highlights:
- Trunk layers (`0 .. block_count - nextn_predict_layers - 1`):
  - Recurrent layers (`(il + 1) % full_attention_interval != 0`):
    Gated DeltaNet with `attn_qkv` + `attn_gate` (or combined `attn_qkvz`), causal depthwise
    1D convolution (`ssm_conv1d.weight`) + SiLU, L2-normalized Q and K (`build_gdn_l2_norm`),
    `ssm_beta` (sigmoid), `ssm_alpha` + `ssm_dt.bias` (softplus) * `ssm_a` decay gate,
    recurrent `(head_v_dim, head_v_dim)` delta-rule state update per value head,
    gated RMSNorm (`ssm_norm.weight` * SiLU(z)), and `ssm_out.weight` output projection.
  - Full-attention layers (`(il + 1) % full_attention_interval == 0`):
    Gated GQA where `attn_q.weight` outputs interleaved `(n_head, 2, head_dim)` query + gate,
    head-wise `attn_q_norm.weight` and `attn_k_norm.weight` RMSNorm, partial half-split RoPE
    on the first `rope_dimension_count` dimensions, scaled dot-product GQA with KV cache,
    elementwise `* sigmoid(gate)`, and `attn_output.weight` projection.
  - Post-attention RMSNorm (`post_attention_norm.weight`) + SwiGLU FFN (`ffn_gate`, `ffn_up`, `ffn_down`).
- Optional Multi-Token Prediction (MTP / NextN) draft layer (`blk.{n_layer}.nextn.*`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.inference.dequant import compute_tensor_bytes, dequantize_bytes, get_block_size
from core.inference.gguf_loader import GGUFLoader
from core.inference.kv_cache import (
    ContextLengthExceededError,
    InsufficientMemoryError,
    get_available_memory_bytes,
)
from core.inference.sampler import Sampler, SamplerConfig
from core.inference.tokenizer import GGUFTokenizer
from core.models.hermes_adapter import Hermes36BAdapter


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm along the last dimension in float32."""
    x_f32 = x.astype(np.float32, copy=False)
    variance = np.mean(np.square(x_f32), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + np.float32(eps))
    return (x_f32 * inv_rms) * weight.astype(np.float32, copy=False)


def gdn_l2_norm(x: np.ndarray, eps: float) -> np.ndarray:
    """Exact match to llama.cpp `build_gdn_l2_norm`:
    `ggml_scale(ggml_rms_norm(x, eps / n), 1.0 / sqrt(n))` == `x / sqrt(sum(x^2) + eps)`.
    """
    x_f32 = x.astype(np.float32, copy=False)
    n = float(x_f32.shape[-1])
    variance = np.mean(np.square(x_f32), axis=-1, keepdims=True)
    rms_normed = x_f32 / np.sqrt(variance + np.float32(eps / n))
    return rms_normed * np.float32(1.0 / np.sqrt(n))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x) in float32."""
    x_f32 = x.astype(np.float32, copy=False)
    clipped = np.clip(x_f32, -88.0, 88.0)
    return x_f32 / (1.0 + np.exp(-clipped))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid in float32."""
    x_f32 = x.astype(np.float32, copy=False)
    clipped = np.clip(x_f32, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x)) in float32."""
    x_f32 = x.astype(np.float32, copy=False)
    return np.where(
        x_f32 > 20.0,
        x_f32,
        np.log1p(np.exp(np.clip(x_f32, -88.0, 20.0))),
    ).astype(np.float32, copy=False)


def apply_partial_half_split_rope(
    x: np.ndarray,
    positions: np.ndarray,
    rope_dim: int,
    freq_base: float,
) -> np.ndarray:
    """Apply half-split (NeoX / IMROPE text-equivalent) RoPE to the first `rope_dim` elements
    of each head in `x` (shape `(seq_len, num_heads, head_dim)`), leaving `rope_dim..head_dim` untouched.
    """
    if rope_dim <= 0:
        return x
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")

    half = rope_dim // 2
    exponents = np.arange(0, rope_dim, 2, dtype=np.float32) / np.float32(rope_dim)
    inv_freq = 1.0 / (np.float32(freq_base) ** exponents)  # (half,)

    pos_f32 = positions.astype(np.float32, copy=False)[:, None]  # (seq_len, 1)
    angles = pos_f32 * inv_freq[None, :]  # (seq_len, half)
    cos = np.cos(angles)[:, None, :]  # (seq_len, 1, half)
    sin = np.sin(angles)[:, None, :]  # (seq_len, 1, half)

    out = x.astype(np.float32, copy=True)
    x1 = out[:, :, :half]
    x2 = out[:, :, half:rope_dim]
    out_first = np.empty((out.shape[0], out.shape[1], rope_dim), dtype=np.float32)
    out_first[:, :, :half] = x1 * cos - x2 * sin
    out_first[:, :, half:rope_dim] = x2 * cos + x1 * sin
    out[:, :, :rope_dim] = out_first
    return out


@dataclass(frozen=True)
class Qwen35Config:
    """Dynamic model hyperparameters read from `qwen35.*` GGUF metadata."""

    architecture: str
    block_count_all: int
    nextn_predict_layers: int
    block_count: int
    context_length: int
    embedding_length: int
    feed_forward_length: int
    head_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    rope_dimension_count: int
    rope_freq_base: float
    rms_norm_eps: float
    full_attention_interval: int
    ssm_conv_kernel: int
    ssm_state_size: int
    ssm_group_count: int
    ssm_time_step_rank: int
    ssm_inner_size: int
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    def is_recurrent_layer(self, layer_idx: int) -> bool:
        """True if `layer_idx` is a Gated DeltaNet recurrent linear-attention layer."""
        return (layer_idx < self.block_count) and (
            (layer_idx + 1) % self.full_attention_interval != 0
        )

    @property
    def recurrent_layer_indices(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.block_count) if self.is_recurrent_layer(i))

    @property
    def full_attention_layer_indices(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.block_count) if not self.is_recurrent_layer(i))

    @property
    def ssm_key_dim(self) -> int:
        return self.ssm_state_size * self.ssm_group_count

    @property
    def ssm_value_dim(self) -> int:
        return self.ssm_inner_size

    @property
    def ssm_head_v_dim(self) -> int:
        return self.ssm_inner_size // self.ssm_time_step_rank

    @property
    def ssm_conv_dim(self) -> int:
        return 2 * self.ssm_key_dim + self.ssm_value_dim

    @classmethod
    def from_gguf_metadata(
        cls,
        metadata: Dict[str, Any],
        tensor_infos: Optional[Dict[str, Any]] = None,
    ) -> "Qwen35Config":
        arch = metadata.get("general.architecture")
        if arch != "qwen35":
            raise ValueError(f"Expected general.architecture='qwen35', got {arch!r}")

        def _req(key: str) -> Any:
            if key not in metadata:
                raise ValueError(f"Missing required GGUF metadata key: '{key}'")
            return metadata[key]

        block_count_all = int(_req("qwen35.block_count"))
        nextn_layers = int(metadata.get("qwen35.nextn_predict_layers", 0))
        block_count = block_count_all - nextn_layers
        if block_count <= 0:
            raise ValueError(
                f"Invalid trunk block_count={block_count} (block_count_all={block_count_all}, nextn={nextn_layers})"
            )

        context_length = int(_req("qwen35.context_length"))
        embedding_length = int(_req("qwen35.embedding_length"))
        feed_forward_length = int(_req("qwen35.feed_forward_length"))
        head_count = int(_req("qwen35.attention.head_count"))
        head_count_kv = int(_req("qwen35.attention.head_count_kv"))
        key_length = int(metadata.get("qwen35.attention.key_length", embedding_length // head_count))
        value_length = int(metadata.get("qwen35.attention.value_length", key_length))
        rope_dim = int(metadata.get("qwen35.rope.dimension_count", key_length))
        rope_freq_base = float(metadata.get("qwen35.rope.freq_base", 10000000.0))
        rms_norm_eps = float(metadata.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6))
        full_attn_interval = int(metadata.get("qwen35.full_attention_interval", 4))

        ssm_conv_kernel = int(metadata.get("qwen35.ssm.conv_kernel", 4))
        ssm_state_size = int(metadata.get("qwen35.ssm.state_size", 128))
        ssm_group_count = int(metadata.get("qwen35.ssm.group_count", 16))
        ssm_time_step_rank = int(metadata.get("qwen35.ssm.time_step_rank", 48))
        ssm_inner_size = int(
            metadata.get("qwen35.ssm.inner_size", ssm_state_size * ssm_time_step_rank)
        )

        tokens = metadata.get("tokenizer.ggml.tokens")
        if isinstance(tokens, list) and len(tokens) > 0:
            vocab_size = len(tokens)
        elif "qwen35.vocab_size" in metadata:
            vocab_size = int(metadata["qwen35.vocab_size"])
        elif tensor_infos and "token_embd.weight" in tensor_infos:
            vocab_size = int(tensor_infos["token_embd.weight"].dims[1])
        else:
            raise ValueError("Cannot determine vocab_size from GGUF metadata")

        bos_id = int(metadata.get("tokenizer.ggml.bos_token_id", 248044))
        eos_id = int(metadata.get("tokenizer.ggml.eos_token_id", 248046))
        pad_id = int(metadata.get("tokenizer.ggml.padding_token_id", 248055))

        return cls(
            architecture=arch,
            block_count_all=block_count_all,
            nextn_predict_layers=nextn_layers,
            block_count=block_count,
            context_length=context_length,
            embedding_length=embedding_length,
            feed_forward_length=feed_forward_length,
            head_count=head_count,
            head_count_kv=head_count_kv,
            key_length=key_length,
            value_length=value_length,
            rope_dimension_count=rope_dim,
            rope_freq_base=rope_freq_base,
            rms_norm_eps=rms_norm_eps,
            full_attention_interval=full_attn_interval,
            ssm_conv_kernel=ssm_conv_kernel,
            ssm_state_size=ssm_state_size,
            ssm_group_count=ssm_group_count,
            ssm_time_step_rank=ssm_time_step_rank,
            ssm_inner_size=ssm_inner_size,
            vocab_size=vocab_size,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )


class Qwen35HybridCache:
    """Hybrid state cache for `qwen35`:
    - Holds recurrent `(conv_state, ssm_state)` for Gated DeltaNet layers.
    - Holds growing KV cache `(k_buf, v_buf)` for Gated Full-Attention layers.
    """

    def __init__(
        self,
        config: Qwen35Config,
        max_seq_len: int,
        dtype: Any = np.float32,
        memory_limit_bytes: Optional[int] = None,
        initial_capacity: int = 64,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.config = config
        self.max_seq_len = int(max_seq_len)
        self.dtype = np.dtype(dtype)

        n_full = len(config.full_attention_layer_indices)
        n_rec = len(config.recurrent_layer_indices)

        # Memory budget check: full KV cache at max_seq_len + recurrent state matrices
        kv_bytes = (
            2
            * n_full
            * config.head_count_kv
            * config.key_length
            * self.max_seq_len
            * self.dtype.itemsize
        )
        conv_state_bytes = (
            n_rec * max(0, config.ssm_conv_kernel - 1) * config.ssm_conv_dim * 4
        )
        ssm_state_bytes = (
            n_rec
            * config.ssm_time_step_rank
            * config.ssm_head_v_dim
            * config.ssm_head_v_dim
            * 4
        )
        self.required_bytes = int(kv_bytes + conv_state_bytes + ssm_state_bytes)

        budget = (
            int(memory_limit_bytes)
            if memory_limit_bytes is not None
            else int(get_available_memory_bytes() * 0.90)
        )
        if self.required_bytes > budget:
            raise InsufficientMemoryError(
                f"Qwen35HybridCache requires {self.required_bytes:,} bytes "
                f"({self.required_bytes / (1024 ** 2):.2f} MiB) for max_seq_len={self.max_seq_len}, "
                f"which exceeds available memory budget of {budget:,} bytes."
            )

        self._seq_len = 0
        self._capacity = min(max(1, int(initial_capacity)), self.max_seq_len)

        # Map full-attention layer index -> KV buffer slot
        self._attn_layer_to_slot: Dict[int, int] = {
            layer_idx: slot
            for slot, layer_idx in enumerate(config.full_attention_layer_indices)
        }
        self._k = np.zeros(
            (n_full, self._capacity, config.head_count_kv, config.key_length),
            dtype=self.dtype,
        )
        self._v = np.zeros(
            (n_full, self._capacity, config.head_count_kv, config.value_length),
            dtype=self.dtype,
        )

        # Recurrent states keyed by layer_idx
        conv_hist = max(0, config.ssm_conv_kernel - 1)
        self.conv_states: Dict[int, np.ndarray] = {
            layer_idx: np.zeros((conv_hist, config.ssm_conv_dim), dtype=np.float32)
            for layer_idx in config.recurrent_layer_indices
        }
        self.ssm_states: Dict[int, np.ndarray] = {
            layer_idx: np.zeros(
                (config.ssm_time_step_rank, config.ssm_head_v_dim, config.ssm_head_v_dim),
                dtype=np.float32,
            )
            for layer_idx in config.recurrent_layer_indices
        }

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def advance_seq_len(self, n_new: int) -> None:
        if self._seq_len + n_new > self.max_seq_len:
            raise ContextLengthExceededError(
                f"Sequence length {self._seq_len + n_new} exceeds max_seq_len={self.max_seq_len}"
            )
        self._seq_len += n_new

    def _ensure_kv_capacity(self, required_len: int) -> None:
        if required_len <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < required_len:
            new_cap = min(new_cap * 2, self.max_seq_len)
        n_full = len(self._attn_layer_to_slot)
        new_k = np.zeros(
            (n_full, new_cap, self.config.head_count_kv, self.config.key_length),
            dtype=self.dtype,
        )
        new_v = np.zeros(
            (n_full, new_cap, self.config.head_count_kv, self.config.value_length),
            dtype=self.dtype,
        )
        if self._seq_len > 0:
            new_k[:, : self._seq_len] = self._k[:, : self._seq_len]
            new_v[:, : self._seq_len] = self._v[:, : self._seq_len]
        self._k = new_k
        self._v = new_v
        self._capacity = new_cap

    def update_full_attention_kv(
        self,
        layer_idx: int,
        k_new: np.ndarray,
        v_new: np.ndarray,
        start_pos: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        slot = self._attn_layer_to_slot[layer_idx]
        n_new = k_new.shape[0]
        end_pos = start_pos + n_new
        if end_pos > self.max_seq_len:
            raise ContextLengthExceededError(
                f"KV cache position {end_pos} exceeds max_seq_len={self.max_seq_len}"
            )
        self._ensure_kv_capacity(end_pos)
        self._k[slot, start_pos:end_pos] = k_new
        self._v[slot, start_pos:end_pos] = v_new
        return self._k[slot, :end_pos], self._v[slot, :end_pos]

    def reset(self) -> None:
        self._seq_len = 0
        self._k.fill(0)
        self._v.fill(0)
        for st in self.conv_states.values():
            st.fill(0)
        for st in self.ssm_states.values():
            st.fill(0)


class Qwen35Engine:
    """Native inference engine for `general.architecture = "qwen35"` (Qwen3.5 / Qwen3.8)."""

    SUPPORTED_ARCHITECTURE = "qwen35"

    def __init__(
        self,
        gguf_path: Union[str, Path],
        *,
        max_context_length: Optional[int] = None,
        preload_weights: Optional[bool] = None,
        memory_limit_bytes: Optional[int] = None,
    ) -> None:
        self.gguf_path = Path(gguf_path)
        self.loader = GGUFLoader(self.gguf_path, header_only=False)
        try:
            self.config = Qwen35Config.from_gguf_metadata(
                self.loader.metadata, self.loader.tensors
            )
            self.max_context_length = (
                int(max_context_length)
                if max_context_length is not None
                else min(self.config.context_length, 4096)
            )
            if self.max_context_length <= 0 or self.max_context_length > self.config.context_length:
                raise ValueError(
                    f"Invalid max_context_length={self.max_context_length} "
                    f"(model context_length={self.config.context_length})"
                )

            self.memory_limit_bytes = memory_limit_bytes
            # Validate hybrid cache memory budget upfront
            _ = self.create_cache(max_seq_len=self.max_context_length)

            self.tokenizer = GGUFTokenizer.from_gguf_metadata(self.loader.metadata)
            self.adapter = Hermes36BAdapter(thinking=False, keep_cots=False)

            # Default preload_weights: True for small synthetic test checkpoints (< 256 MiB),
            # False for multi-gigabyte production checkpoints so they stream via mmap.
            if preload_weights is None:
                preload_weights = self.loader.file_size < (256 * 1024 * 1024)
            self.preload_weights = bool(preload_weights)
            self._weight_cache: Dict[str, np.ndarray] = {}
            self._active_sessions: int = 0

            # Always preload small 1D F32 tensors (norms, ssm_a, ssm_dt.bias, ssm_conv1d)
            self._preload_small_tensors()
            if self.preload_weights:
                self._preload_all_weights()
        except Exception:
            self.loader.close()
            raise

    @property
    def active_sessions(self) -> int:
        return self._active_sessions

    def _preload_small_tensors(self) -> None:
        for name, info in self.loader.tensors.items():
            if info.n_bytes <= 1024 * 1024:  # <= 1 MiB
                self._weight_cache[name] = self.loader.get_tensor_f32(name)

    def _preload_all_weights(self) -> None:
        for name in self.loader.tensors:
            if name not in self._weight_cache:
                self._weight_cache[name] = self.loader.get_tensor_f32(name)

    def _get_weight(self, name: str) -> np.ndarray:
        cached = self._weight_cache.get(name)
        if cached is not None:
            return cached
        return self.loader.get_tensor_f32(name)

    def _has_weight(self, name: str) -> bool:
        return name in self.loader.tensors

    def _embed_tokens(self, token_ids: Sequence[int]) -> np.ndarray:
        if "token_embd.weight" in self._weight_cache:
            return self._weight_cache["token_embd.weight"][list(token_ids)].astype(
                np.float32, copy=True
            )
        return self.loader.get_tensor_rows_f32("token_embd.weight", token_ids)

    def _matmul_weight(self, x: np.ndarray, weight_name: str, chunk_rows: int = 16384) -> np.ndarray:
        """Compute `x @ W.T` in float32. For very large matrices (such as `output.weight`
        with 248,320 rows), streams dequantization in chunks of `chunk_rows` rows so
        peak RAM stays bounded.
        """
        if weight_name in self._weight_cache:
            w = self._weight_cache[weight_name]
            return np.matmul(x, w.T)

        info = self.loader.get_tensor_info(weight_name)
        if len(info.dims) != 2:
            w = self.loader.get_tensor_f32(weight_name)
            return np.matmul(x, w.T)

        in_dim, out_rows = int(info.dims[0]), int(info.dims[1])
        if out_rows <= chunk_rows or self.loader._mmap is None:
            w = self.loader.get_tensor_f32(weight_name)
            return np.matmul(x, w.T)

        # Chunked streaming matmul directly from mmap buffer
        row_bytes = compute_tensor_bytes(info.type_id, in_dim)
        base_offset = self.loader.data_offset + info.offset
        mm = self.loader._mmap

        out_shape = (*x.shape[:-1], out_rows)
        result = np.empty(out_shape, dtype=np.float32)

        for start_row in range(0, out_rows, chunk_rows):
            end_row = min(start_row + chunk_rows, out_rows)
            n_rows = end_row - start_row
            byte_start = base_offset + start_row * row_bytes
            byte_end = byte_start + n_rows * row_bytes
            raw_chunk = mm[byte_start:byte_end]
            w_chunk = dequantize_bytes(raw_chunk, info.type_id, n_rows * in_dim).reshape(
                (n_rows, in_dim)
            )
            result[..., start_row:end_row] = np.matmul(x, w_chunk.T)

        return result

    def create_cache(self, max_seq_len: Optional[int] = None) -> Qwen35HybridCache:
        limit_seq = int(max_seq_len) if max_seq_len is not None else self.max_context_length
        return Qwen35HybridCache(
            config=self.config,
            max_seq_len=limit_seq,
            dtype=np.float32,
            memory_limit_bytes=self.memory_limit_bytes,
        )

    def _forward_recurrent_layer(
        self,
        h: np.ndarray,
        layer_idx: int,
        cache: Qwen35HybridCache,
        layer_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    ) -> np.ndarray:
        """Execute one Gated DeltaNet linear-attention layer (`qwen35.cpp::build_layer_attn_linear`)."""
        cfg = self.config
        prefix = f"blk.{layer_idx}"
        seq_len = h.shape[0]

        num_k_heads = cfg.ssm_group_count
        head_k_dim = cfg.ssm_state_size
        num_v_heads = cfg.ssm_time_step_rank
        head_v_dim = cfg.ssm_head_v_dim
        key_dim = cfg.ssm_key_dim
        value_dim = cfg.ssm_value_dim
        conv_dim = cfg.ssm_conv_dim

        # 1) Input projections: qkv_mixed (seq_len, conv_dim) and z (seq_len, value_dim)
        if self._has_weight(f"{prefix}.attn_qkvz.weight"):
            qkvz = self._matmul_weight(h, f"{prefix}.attn_qkvz.weight")
            qkv_mixed = qkvz[:, :conv_dim]
            z = qkvz[:, conv_dim:]
        else:
            qkv_mixed = self._matmul_weight(h, f"{prefix}.attn_qkv.weight")
            z = self._matmul_weight(h, f"{prefix}.attn_gate.weight")

        if layer_callback:
            layer_callback(f"{prefix}.qkv_mixed", qkv_mixed)

        # 2) Beta & Alpha gates
        if self._has_weight(f"{prefix}.ssm_ba.weight"):
            ba = self._matmul_weight(h, f"{prefix}.ssm_ba.weight")
            beta_raw = ba[:, :num_v_heads]
            alpha_raw = ba[:, num_v_heads:]
        else:
            beta_raw = self._matmul_weight(h, f"{prefix}.ssm_beta.weight")
            alpha_raw = self._matmul_weight(h, f"{prefix}.ssm_alpha.weight")

        beta = sigmoid(beta_raw)  # (seq_len, num_v_heads)
        dt_bias = self._get_weight(f"{prefix}.ssm_dt.bias")  # (num_v_heads,)
        ssm_a = self._get_weight(f"{prefix}.ssm_a")  # (num_v_heads,)
        alpha_softplus = softplus(alpha_raw + dt_bias)
        gate = alpha_softplus * ssm_a  # (seq_len, num_v_heads)

        # 3) Depthwise causal 1D convolution over qkv_mixed with conv_state
        conv_w = self._get_weight(f"{prefix}.ssm_conv1d.weight")  # shape (conv_dim, d_conv)
        d_conv = cfg.ssm_conv_kernel
        if conv_w.shape == (d_conv, conv_dim):
            conv_w = conv_w.T  # ensure (conv_dim, d_conv)

        conv_state = cache.conv_states[layer_idx]  # (d_conv - 1, conv_dim)
        if d_conv > 1:
            conv_input = np.vstack([conv_state, qkv_mixed])  # (d_conv - 1 + seq_len, conv_dim)
            cache.conv_states[layer_idx] = conv_input[seq_len : seq_len + d_conv - 1].copy()
        else:
            conv_input = qkv_mixed

        conv_out = np.zeros((seq_len, conv_dim), dtype=np.float32)
        for k_idx in range(d_conv):
            conv_out += conv_input[k_idx : k_idx + seq_len] * conv_w[:, k_idx]

        conv_silu = silu(conv_out)

        # 4) Split Q, K, V and L2-normalize Q, K
        q_conv = conv_silu[:, :key_dim].reshape(seq_len, num_k_heads, head_k_dim)
        k_conv = conv_silu[:, key_dim : 2 * key_dim].reshape(seq_len, num_k_heads, head_k_dim)
        v_conv = conv_silu[:, 2 * key_dim :].reshape(seq_len, num_v_heads, head_v_dim)

        q_conv = gdn_l2_norm(q_conv, cfg.rms_norm_eps)
        k_conv = gdn_l2_norm(k_conv, cfg.rms_norm_eps)

        # 5) Gated DeltaNet recurrent step across tokens (vectorized across all num_v_heads)
        # Note from ggml_compute_forward_gated_delta_net_one_chunk:
        # iq1 = iv1 % neq1, ik1 = iv1 % nek1
        head_map = np.arange(num_v_heads) % num_k_heads
        state = cache.ssm_states[layer_idx]  # (num_v_heads, head_v_dim, head_v_dim)
        scale = np.float32(1.0 / np.sqrt(float(head_v_dim)))

        attn_output = np.empty((seq_len, num_v_heads, head_v_dim), dtype=np.float32)
        for t in range(seq_len):
            q_t = q_conv[t, head_map]  # (num_v_heads, head_v_dim)
            k_t = k_conv[t, head_map]  # (num_v_heads, head_v_dim)
            v_t = v_conv[t]            # (num_v_heads, head_v_dim)

            decay_t = np.exp(gate[t])[:, None, None]  # (num_v_heads, 1, 1)
            beta_t = beta[t][:, None]                 # (num_v_heads, 1)

            state *= decay_t
            # delta[h, j] = (v_t[h, j] - sum_i state[h, j, i] * k_t[h, i]) * beta_t[h]
            sk = np.einsum("hji,hi->hj", state, k_t)
            delta = (v_t - sk) * beta_t
            # state[h, j, i] += delta[h, j] * k_t[h, i]
            state += delta[:, :, None] * k_t[:, None, :]
            # out[h, j] = sum_i state[h, j, i] * q_t[h, i] * scale
            out_t = np.einsum("hji,hi->hj", state, q_t) * scale
            attn_output[t] = out_t

        # 6) Gated RMSNorm (`build_norm_gated`) & Output Projection (`ssm_out.weight`)
        ssm_norm_w = self._get_weight(f"{prefix}.ssm_norm.weight")  # (head_v_dim,)
        z_heads = z.reshape(seq_len, num_v_heads, head_v_dim)
        normed = rms_norm(attn_output, ssm_norm_w, cfg.rms_norm_eps) * silu(z_heads)
        flat_out = normed.reshape(seq_len, value_dim)

        return self._matmul_weight(flat_out, f"{prefix}.ssm_out.weight")

    def _forward_full_attention_layer(
        self,
        h: np.ndarray,
        layer_idx: int,
        positions: np.ndarray,
        start_pos: int,
        cache: Qwen35HybridCache,
        layer_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    ) -> np.ndarray:
        """Execute one Gated Full-Attention layer (`qwen35.cpp::build_layer_attn`)."""
        cfg = self.config
        prefix = f"blk.{layer_idx}"
        seq_len = h.shape[0]
        n_head = cfg.head_count
        n_kv = cfg.head_count_kv
        head_dim = cfg.key_length

        # 1) Joint Q+Gate projection: shape (seq_len, n_head * head_dim * 2)
        q_full = self._matmul_weight(h, f"{prefix}.attn_q.weight")
        qg = q_full.reshape(seq_len, n_head, 2, head_dim)
        q = qg[:, :, 0, :]  # (seq_len, n_head, head_dim)
        gate = qg[:, :, 1, :].reshape(seq_len, n_head * head_dim)  # (seq_len, n_head * head_dim)

        k = self._matmul_weight(h, f"{prefix}.attn_k.weight").reshape(seq_len, n_kv, head_dim)
        v = self._matmul_weight(h, f"{prefix}.attn_v.weight").reshape(seq_len, n_kv, head_dim)

        # 2) Head-wise Q and K RMSNorm
        q_norm_w = self._get_weight(f"{prefix}.attn_q_norm.weight")
        k_norm_w = self._get_weight(f"{prefix}.attn_k_norm.weight")
        q = rms_norm(q, q_norm_w, cfg.rms_norm_eps)
        k = rms_norm(k, k_norm_w, cfg.rms_norm_eps)

        # 3) Partial RoPE on first `rope_dimension_count` dimensions
        q = apply_partial_half_split_rope(q, positions, cfg.rope_dimension_count, cfg.rope_freq_base)
        k = apply_partial_half_split_rope(k, positions, cfg.rope_dimension_count, cfg.rope_freq_base)

        # 4) Update KV cache for this full-attention layer
        k_total, v_total = cache.update_full_attention_kv(layer_idx, k, v, start_pos=start_pos)
        total_len = k_total.shape[0]

        # 5) Grouped-Query Attention (GQA)
        group_size = n_head // n_kv
        q_grouped = q.reshape(seq_len, n_kv, group_size, head_dim).transpose(1, 2, 0, 3)
        k_trans = k_total.transpose(1, 2, 0)
        scale = np.float32(1.0 / np.sqrt(float(head_dim)))
        scores = np.matmul(q_grouped, k_trans[:, None, :, :]) * scale

        if seq_len > 1:
            q_pos = positions[:, None]
            k_pos = np.arange(total_len, dtype=np.int64)[None, :]
            causal_mask = k_pos > q_pos
            scores = np.where(causal_mask[None, None, :, :], np.float32(-1e9), scores)

        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        v_trans = v_total.transpose(1, 0, 2)
        attn_out = np.matmul(probs, v_trans[:, None, :, :])
        attn_pregate = (
            attn_out.transpose(2, 0, 1, 3).reshape(seq_len, n_head * head_dim)
        )

        # 6) Output gate sigmoid multiplication & output projection
        attn_gated = attn_pregate * sigmoid(gate)
        if layer_callback:
            layer_callback(f"{prefix}.attn_gated", attn_gated)

        return self._matmul_weight(attn_gated, f"{prefix}.attn_output.weight")

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        kv_cache: Optional[Qwen35HybridCache] = None,
        start_pos: Optional[int] = None,
        max_layers: Optional[int] = None,
        return_all_logits: bool = True,
        layer_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    ) -> np.ndarray:
        """Run the `qwen35` forward pass over `token_ids`.

        Parameters:
        - `max_layers`: Optional cap on number of trunk layers to execute (useful for fast
          layer-subset smoke verification on 27B checkpoints). Defaults to all `config.block_count` layers.
        - `return_all_logits`: If False, projects only the final token position through `output.weight`.
        """
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        seq_len = len(token_ids)
        for tid in token_ids:
            if tid < 0 or tid >= self.config.vocab_size:
                raise ValueError(
                    f"Token ID {tid} out of range [0, {self.config.vocab_size})"
                )

        cache = kv_cache if kv_cache is not None else self.create_cache()
        pos0 = cache.seq_len if start_pos is None else int(start_pos)
        if pos0 + seq_len > self.max_context_length:
            raise ContextLengthExceededError(
                f"Prompt length {pos0 + seq_len} exceeds max_context_length={self.max_context_length}"
            )

        positions = np.arange(pos0, pos0 + seq_len, dtype=np.int64)
        x = self._embed_tokens(token_ids)  # (seq_len, hidden_size)

        n_layers = (
            min(int(max_layers), self.config.block_count)
            if max_layers is not None
            else self.config.block_count
        )

        for il in range(n_layers):
            prefix = f"blk.{il}"
            residual = x
            attn_norm_w = self._get_weight(f"{prefix}.attn_norm.weight")
            h = rms_norm(x, attn_norm_w, self.config.rms_norm_eps)

            if self.config.is_recurrent_layer(il):
                attn_out = self._forward_recurrent_layer(h, il, cache, layer_callback)
            else:
                attn_out = self._forward_full_attention_layer(
                    h, il, positions, pos0, cache, layer_callback
                )

            x = residual + attn_out

            # Post-attention RMSNorm + SwiGLU FFN
            ffn_residual = x
            post_norm_w = self._get_weight(f"{prefix}.post_attention_norm.weight")
            h_ffn = rms_norm(x, post_norm_w, self.config.rms_norm_eps)

            gate_ffn = silu(self._matmul_weight(h_ffn, f"{prefix}.ffn_gate.weight"))
            up_ffn = self._matmul_weight(h_ffn, f"{prefix}.ffn_up.weight")
            ffn_out = self._matmul_weight(gate_ffn * up_ffn, f"{prefix}.ffn_down.weight")
            x = ffn_residual + ffn_out

            if layer_callback:
                layer_callback(f"{prefix}.out", x)

        cache.advance_seq_len(seq_len)

        out_norm_w = self._get_weight("output_norm.weight")
        x_norm = rms_norm(x, out_norm_w, self.config.rms_norm_eps)

        out_weight_name = (
            "output.weight" if self._has_weight("output.weight") else "token_embd.weight"
        )
        if not return_all_logits and seq_len > 1:
            last_logits = self._matmul_weight(x_norm[-1:], out_weight_name)
            return last_logits.astype(np.float32, copy=False)

        logits = self._matmul_weight(x_norm, out_weight_name)
        return logits.astype(np.float32, copy=False)

    def generate_stream(
        self,
        prompt: Union[str, Sequence[int]],
        *,
        sampler_config: Optional[SamplerConfig] = None,
        max_new_tokens: int = 64,
        stop_sequences: Optional[Sequence[str]] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        add_bos: bool = False,
        cancel_callback: Optional[Callable[[int], bool]] = None,
        max_layers: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Autoregressively stream generated tokens using `Qwen35HybridCache`."""
        if isinstance(prompt, str):
            if len(prompt) == 0:
                raise ValueError("Prompt string cannot be empty")
            prompt_ids = self.tokenizer.encode(prompt, add_bos=add_bos)
        else:
            prompt_ids = [int(t) for t in prompt]

        if not prompt_ids:
            raise ValueError("Prompt token_ids cannot be empty")

        total_needed = len(prompt_ids) + max(0, int(max_new_tokens))
        if total_needed > self.max_context_length:
            raise ContextLengthExceededError(
                f"Prompt ({len(prompt_ids)} tokens) + max_new_tokens ({max_new_tokens}) "
                f"exceeds max_context_length={self.max_context_length}"
            )

        sampler = Sampler(sampler_config or SamplerConfig(temperature=0.0))
        cache = self.create_cache(max_seq_len=self.max_context_length)

        eos_ids = {self.config.eos_token_id}
        if stop_token_ids is not None:
            eos_ids.update(int(t) for t in stop_token_ids)
        stop_strs = [s for s in (stop_sequences or []) if s]

        self._active_sessions += 1
        try:
            logits = self.forward(
                prompt_ids,
                kv_cache=cache,
                max_layers=max_layers,
                return_all_logits=False,
            )
            next_logits = logits[-1]

            generated_ids: List[int] = []
            emitted_text_len = 0

            for step in range(max_new_tokens):
                if cancel_callback is not None and not cancel_callback(step):
                    break

                next_token = sampler.sample(
                    next_logits,
                    previous_tokens=prompt_ids + generated_ids,
                )
                generated_ids.append(next_token)
                full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                delta_text = full_text[emitted_text_len:]
                emitted_text_len = len(full_text)

                if next_token in eos_ids:
                    yield {
                        "token_id": next_token,
                        "text": delta_text,
                        "step": step,
                        "finish_reason": "eos",
                    }
                    break

                matched_stop: Optional[str] = None
                for s in stop_strs:
                    if s in full_text:
                        matched_stop = s
                        break

                if matched_stop is not None:
                    stop_idx = full_text.find(matched_stop)
                    prev_len = emitted_text_len - len(delta_text)
                    trimmed_delta = (
                        full_text[prev_len:stop_idx] if stop_idx >= prev_len else ""
                    )
                    yield {
                        "token_id": next_token,
                        "text": trimmed_delta,
                        "step": step,
                        "finish_reason": "stop_sequence",
                        "stop_sequence": matched_stop,
                    }
                    break

                finish_reason = "length" if step == max_new_tokens - 1 else None
                yield {
                    "token_id": next_token,
                    "text": delta_text,
                    "step": step,
                    "finish_reason": finish_reason,
                }

                if step < max_new_tokens - 1:
                    step_logits = self.forward(
                        [next_token],
                        kv_cache=cache,
                        max_layers=max_layers,
                        return_all_logits=False,
                    )
                    next_logits = step_logits[-1]
        finally:
            self._active_sessions -= 1
            cache.reset()

    def generate(
        self,
        prompt: Union[str, Sequence[int]],
        *,
        sampler_config: Optional[SamplerConfig] = None,
        max_new_tokens: int = 64,
        stop_sequences: Optional[Sequence[str]] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        add_bos: bool = False,
        cancel_callback: Optional[Callable[[int], bool]] = None,
        max_layers: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(prompt, str):
            if len(prompt) == 0:
                raise ValueError("Prompt string cannot be empty")
            prompt_ids = self.tokenizer.encode(prompt, add_bos=add_bos)
        else:
            prompt_ids = [int(t) for t in prompt]
            if not prompt_ids:
                raise ValueError("Prompt token_ids cannot be empty")

        token_ids: List[int] = []
        pieces: List[str] = []
        finish_reason: Optional[str] = "length"
        matched_stop: Optional[str] = None

        for ev in self.generate_stream(
            prompt_ids,
            sampler_config=sampler_config,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
            stop_token_ids=stop_token_ids,
            add_bos=False,
            cancel_callback=cancel_callback,
            max_layers=max_layers,
        ):
            token_ids.append(ev["token_id"])
            if ev["finish_reason"] != "eos":
                pieces.append(ev["text"])
            if ev["finish_reason"] is not None:
                finish_reason = ev["finish_reason"]
                matched_stop = ev.get("stop_sequence")

        return {
            "prompt_token_ids": prompt_ids,
            "token_ids": token_ids,
            "text": "".join(pieces),
            "finish_reason": finish_reason,
            "stop_sequence": matched_stop,
        }

    @staticmethod
    def format_qwen_chat_prompt(
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages using the Qwen ChatML (`<|im_start|>...<|im_end|>`) template."""
        if not messages:
            raise ValueError("messages list cannot be empty")

        parts: List[str] = []
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            sys_text = "You are Qwen, a helpful AI assistant."
            if tools:
                import json as _json

                sys_text += (
                    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n"
                    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                    + "\n".join(_json.dumps(t) for t in tools)
                    + '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
                )
            parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>\n")

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""
            if role == "tool":
                parts.append(
                    f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            if thinking:
                parts.append("<|im_start|>assistant\n<think>\n")
            else:
                parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
        keep_cots: bool = False,
        sampler_config: Optional[SamplerConfig] = None,
        max_new_tokens: int = 64,
        stop_sequences: Optional[Sequence[str]] = None,
        max_layers: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not messages:
            raise ValueError("messages list cannot be empty")

        prompt_str = self.format_qwen_chat_prompt(
            messages, tools=tools, thinking=thinking, add_generation_prompt=True
        )
        default_stops = ["<|im_end|>", "<|endoftext|>"]
        if stop_sequences:
            for s in stop_sequences:
                if s not in default_stops:
                    default_stops.append(s)

        gen = self.generate(
            prompt_str,
            sampler_config=sampler_config,
            max_new_tokens=max_new_tokens,
            stop_sequences=default_stops,
            add_bos=False,
            max_layers=max_layers,
        )
        raw_text = gen["text"]
        if thinking and not raw_text.lstrip().startswith("<think>"):
            raw_for_parse = "<think>\n" + raw_text
        else:
            raw_for_parse = raw_text

        adapter = Hermes36BAdapter(thinking=thinking, keep_cots=keep_cots)
        parsed = adapter.parse_generation(raw_for_parse)

        return {
            "prompt": prompt_str,
            "prompt_token_ids": gen["prompt_token_ids"],
            "token_ids": gen["token_ids"],
            "generated_token_ids": gen["token_ids"],
            "raw_output": raw_text,
            "finish_reason": gen["finish_reason"],
            "thinking": parsed["thinking"],
            "has_thinking": parsed["has_thinking"],
            "unclosed_thinking": parsed["unclosed_thinking"],
            "content": parsed["final_content"],
            "tool_calls": parsed["tool_calls"],
            "has_tool_calls": parsed["has_tool_calls"],
            "malformed_tool_calls": parsed["malformed_tool_calls"],
            "openai_tool_calls": parsed["openai_tool_calls"],
            "anthropic_tool_use": parsed["anthropic_tool_use"],
        }

    def close(self) -> None:
        self._weight_cache.clear()
        self.loader.close()

    def __enter__(self) -> "Qwen35Engine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


Qwen35ForCausalLM = Qwen35Engine
