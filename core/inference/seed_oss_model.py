"""
From-scratch NumPy implementation of SeedOssForCausalLM (`general.architecture = "seed_oss"`).

Executes real GGUF v2/v3 weights (Q4_K, Q6_K, F32, F16, Q8_0, etc.) for ByteDance Seed-OSS /
NousResearch Hermes-4.3-36B using only the Python standard library and NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.inference.dequant import compute_tensor_bytes, dequantize_bytes
from core.inference.gguf_loader import GGUFLoader
from core.inference.kv_cache import (
    ContextLengthExceededError,
    InsufficientMemoryError,
    KVCache,
    get_available_memory_bytes,
)
from core.inference.sampler import Sampler, SamplerConfig
from core.inference.tokenizer import GGUFTokenizer
from core.models.hermes_adapter import Hermes36BAdapter


class UnsupportedArchitectureError(ValueError):
    """Raised when a GGUF file's `general.architecture` is not supported by SeedOssEngine."""


@dataclass(frozen=True)
class SeedOssConfig:
    """
    Model configuration read 100% dynamically from GGUF header metadata.
    """

    architecture: str
    name: str
    block_count: int
    context_length: int
    embedding_length: int
    feed_forward_length: int
    head_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    rope_freq_base: float
    rms_norm_eps: float
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    @classmethod
    def from_gguf_loader(cls, loader: GGUFLoader) -> "SeedOssConfig":
        meta = loader.metadata
        arch = meta.get("general.architecture")
        if not isinstance(arch, str) or not arch.strip():
            raise ValueError("Missing or empty 'general.architecture' in GGUF metadata.")
        if arch != "seed_oss":
            raise UnsupportedArchitectureError(
                f"Unsupported GGUF architecture {arch!r}; SeedOssEngine requires 'seed_oss'."
            )

        required_keys = [
            f"{arch}.block_count",
            f"{arch}.context_length",
            f"{arch}.embedding_length",
            f"{arch}.feed_forward_length",
            f"{arch}.attention.head_count",
        ]
        for req_key in required_keys:
            if req_key not in meta:
                raise ValueError(f"Missing required GGUF metadata key: {req_key!r}")

        block_count = int(meta[f"{arch}.block_count"])
        context_length = int(meta[f"{arch}.context_length"])
        embedding_length = int(meta[f"{arch}.embedding_length"])
        feed_forward_length = int(meta[f"{arch}.feed_forward_length"])
        head_count = int(meta[f"{arch}.attention.head_count"])
        head_count_kv = int(meta.get(f"{arch}.attention.head_count_kv", head_count))
        key_length = int(meta.get(f"{arch}.attention.key_length", embedding_length // head_count))
        value_length = int(meta.get(f"{arch}.attention.value_length", key_length))
        rope_freq_base = float(meta.get(f"{arch}.rope.freq_base", 10000000.0))
        rms_norm_eps = float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))

        if "token_embd.weight" in loader.tensors:
            vocab_size = int(loader.get_tensor_info("token_embd.weight").numpy_shape[0])
        elif isinstance(meta.get("tokenizer.ggml.tokens"), list):
            vocab_size = len(meta["tokenizer.ggml.tokens"])
        else:
            raise ValueError("Cannot determine vocab_size from GGUF tensors or tokenizer metadata.")

        bos_token_id = int(meta.get("tokenizer.ggml.bos_token_id", 0))
        eos_token_id = int(meta.get("tokenizer.ggml.eos_token_id", 2))
        pad_token_id = int(meta.get("tokenizer.ggml.padding_token_id", 1))
        name = str(meta.get("general.name", "SeedOss"))

        if block_count <= 0 or embedding_length <= 0 or head_count <= 0 or head_count_kv <= 0:
            raise ValueError("Invalid non-positive architectural dimensions in GGUF metadata.")
        if head_count % head_count_kv != 0:
            raise ValueError(
                f"head_count ({head_count}) must be divisible by head_count_kv ({head_count_kv})"
            )

        return cls(
            architecture=arch,
            name=name,
            block_count=block_count,
            context_length=context_length,
            embedding_length=embedding_length,
            feed_forward_length=feed_forward_length,
            head_count=head_count,
            head_count_kv=head_count_kv,
            key_length=key_length,
            value_length=value_length,
            rope_freq_base=rope_freq_base,
            rms_norm_eps=rms_norm_eps,
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )


class SeedOssEngine:
    """
    Native NumPy forward-pass and streaming inference engine for `SeedOssForCausalLM`.
    """

    def __init__(
        self,
        gguf_path_or_loader: Union[str, Path, GGUFLoader],
        *,
        max_context_length: Optional[int] = None,
        memory_limit_bytes: Optional[int] = None,
        preload_weights: Optional[bool] = None,
        header_only: bool = False,
    ) -> None:
        if isinstance(gguf_path_or_loader, GGUFLoader):
            self.loader = gguf_path_or_loader
            self._owns_loader = False
        else:
            self.loader = GGUFLoader(gguf_path_or_loader, header_only=header_only)
            self._owns_loader = True

        self.config = SeedOssConfig.from_gguf_loader(self.loader)
        self.memory_limit_bytes = memory_limit_bytes

        avail_bytes = (
            int(memory_limit_bytes)
            if memory_limit_bytes is not None
            else get_available_memory_bytes()
        )

        if max_context_length is not None:
            req_ctx = int(max_context_length)
            if req_ctx <= 0:
                raise ValueError(f"max_context_length must be > 0, got {req_ctx}")
            if req_ctx > self.config.context_length:
                raise ContextLengthExceededError(
                    f"Requested max_context_length={req_ctx} exceeds model metadata "
                    f"context_length={self.config.context_length}"
                )
            # Validate KV cache memory budget upfront
            req_kv_bytes = (
                2
                * self.config.block_count
                * self.config.head_count_kv
                * self.config.key_length
                * req_ctx
                * 4
            )
            if req_kv_bytes > avail_bytes:
                raise InsufficientMemoryError(
                    f"Requested max_context_length={req_ctx} requires {req_kv_bytes} bytes "
                    f"for KV cache, exceeding available memory budget of {avail_bytes} bytes."
                )
            self.max_context_length = req_ctx
        else:
            full_kv_bytes = (
                2
                * self.config.block_count
                * self.config.head_count_kv
                * self.config.key_length
                * self.config.context_length
                * 4
            )
            if full_kv_bytes <= avail_bytes:
                self.max_context_length = self.config.context_length
            else:
                self.max_context_length = min(self.config.context_length, 4096)

        self.tokenizer: Optional[GGUFTokenizer] = None
        if (
            "tokenizer.ggml.tokens" in self.loader.metadata
            and "tokenizer.ggml.merges" in self.loader.metadata
        ):
            self.tokenizer = GGUFTokenizer.from_gguf_metadata(self.loader.metadata)

        # Precompute RoPE inverse frequencies in float32 matching SeedOssRotaryEmbedding
        dim = self.config.key_length
        exponents = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
        self._inv_freq = (
            np.float32(1.0) / (np.float32(self.config.rope_freq_base) ** exponents)
        ).astype(np.float32)

        # Automatic weight caching policy: cache tensors in memory for compact models (<256 MiB),
        # or stream on-demand from mmap for multi-gigabyte checkpoints.
        if preload_weights is None:
            self._cache_weights = self.loader.file_size < (256 * 1024 * 1024)
        else:
            self._cache_weights = bool(preload_weights)

        self._tensor_cache: Dict[str, np.ndarray] = {}
        self.active_sessions: int = 0

    def create_kv_cache(
        self,
        max_seq_len: Optional[int] = None,
        memory_limit_bytes: Optional[int] = None,
    ) -> KVCache:
        """Create a KVCache configured for this model's layer and GQA dimensions."""
        seq_limit = self.max_context_length if max_seq_len is None else int(max_seq_len)
        if seq_limit > self.config.context_length:
            raise ContextLengthExceededError(
                f"Requested KV cache max_seq_len={seq_limit} exceeds model metadata "
                f"context_length={self.config.context_length}"
            )
        mem_limit = (
            self.memory_limit_bytes if memory_limit_bytes is None else int(memory_limit_bytes)
        )
        return KVCache(
            num_layers=self.config.block_count,
            num_kv_heads=self.config.head_count_kv,
            head_dim=self.config.key_length,
            max_seq_len=seq_limit,
            dtype=np.float32,
            memory_limit_bytes=mem_limit,
        )

    def _get_tensor(self, name: str) -> np.ndarray:
        if self._cache_weights:
            cached = self._tensor_cache.get(name)
            if cached is not None:
                return cached
            arr = self.loader.get_tensor_f32(name)
            self._tensor_cache[name] = arr
            return arr
        return self.loader.get_tensor_f32(name)

    def _get_optional_tensor(self, name: str) -> Optional[np.ndarray]:
        if name not in self.loader.tensors:
            return None
        return self._get_tensor(name)

    def _lookup_embeddings(self, token_ids: Sequence[int]) -> np.ndarray:
        for tid in token_ids:
            if tid < 0 or tid >= self.config.vocab_size:
                raise ValueError(
                    f"Token ID {tid} out of vocabulary range [0, {self.config.vocab_size})"
                )
        if self._cache_weights:
            emb_table = self._get_tensor("token_embd.weight")
            return emb_table[list(token_ids)].astype(np.float32, copy=True)
        return self.loader.get_tensor_rows_f32("token_embd.weight", token_ids)

    @staticmethod
    def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        """
        SeedOssRMSNorm:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * rsqrt(variance + eps)
        return weight * x
        """
        x_f32 = x.astype(np.float32, copy=False)
        variance = np.mean(np.square(x_f32), axis=-1, keepdims=True, dtype=np.float32)
        inv_rms = np.float32(1.0) / np.sqrt(variance + np.float32(eps))
        return (x_f32 * inv_rms) * weight

    def _compute_rope_embeddings(
        self, position_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (cos, sin) tables of shape `(seq_len, key_length)` for `position_ids`.
        """
        pos_f32 = position_ids.astype(np.float32)[:, None]
        freqs = pos_f32 * self._inv_freq[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb).astype(np.float32)
        sin = np.sin(emb).astype(np.float32)
        return cos, sin

    @staticmethod
    def _rotate_half(x: np.ndarray) -> np.ndarray:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return np.concatenate([-x2, x1], axis=-1)

    @classmethod
    def _apply_rotary_pos_emb(
        cls,
        q: np.ndarray,
        k: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply half-split Rotary Position Embedding to `q` and `k` of shape `(heads, seq_len, head_dim)`.
        """
        cos_b = cos[None, :, :]
        sin_b = sin[None, :, :]
        q_embed = (q * cos_b) + (cls._rotate_half(q) * sin_b)
        k_embed = (k * cos_b) + (cls._rotate_half(k) * sin_b)
        return q_embed.astype(np.float32, copy=False), k_embed.astype(np.float32, copy=False)

    @staticmethod
    def _repeat_kv(hidden_states: np.ndarray, n_rep: int) -> np.ndarray:
        """
        Repeat KV heads from `(num_kv_heads, seq_len, head_dim)` to `(num_heads, seq_len, head_dim)`.
        """
        if n_rep == 1:
            return hidden_states
        return np.repeat(hidden_states, n_rep, axis=0)

    @staticmethod
    def _linear(
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out = np.matmul(x, weight.T).astype(np.float32, copy=False)
        if bias is not None:
            out = out + bias
        return out

    def _matmul_weight(
        self,
        x: np.ndarray,
        weight_name: str,
        bias_name: Optional[str] = None,
        chunk_rows: int = 16384,
    ) -> np.ndarray:
        """Compute `x @ W.T (+ bias)` in float32, streaming row chunks directly
        from `mmap` when weights are not preloaded and `out_rows > chunk_rows`.
        """
        bias = self._get_optional_tensor(bias_name) if bias_name else None
        if self._cache_weights:
            w = self._get_tensor(weight_name)
            return self._linear(x, w, bias)

        info = self.loader.get_tensor_info(weight_name)
        if len(info.dims) != 2 or self.loader._mmap is None:
            w = self.loader.get_tensor_f32(weight_name)
            return self._linear(x, w, bias)

        in_dim, out_rows = int(info.dims[0]), int(info.dims[1])
        if out_rows <= chunk_rows:
            w = self.loader.get_tensor_f32(weight_name)
            return self._linear(x, w, bias)

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

        if bias is not None:
            result = result + bias
        return result

    def forward(
        self,
        token_ids: Sequence[int],
        kv_cache: Optional[KVCache] = None,
        start_pos: Optional[int] = None,
        last_token_only: bool = False,
        layer_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> np.ndarray:
        """
        Execute a full SeedOssForCausalLM forward pass over `token_ids`.

        Parameters
        ----------
        token_ids : Sequence[int]
            Input token IDs of length `seq_len >= 1`.
        kv_cache : KVCache, optional
            Optional KV cache for incremental prefill / autoregressive decoding.
        start_pos : int, optional
            Starting sequence index. Defaults to `kv_cache.seq_len` if `kv_cache` is provided, else `0`.
        last_token_only : bool
            If True, only project and return the final token's logits of shape `(vocab_size,)`.
            If False (default), return all positions' logits of shape `(seq_len, vocab_size)`.
        layer_callback : Callable[[int, np.ndarray], None], optional
            Optional hook called at the end of each layer with `(layer_idx, hidden_states)`.
        """
        if not isinstance(token_ids, (list, tuple, np.ndarray)):
            token_ids = list(token_ids)
        seq_len = len(token_ids)
        if seq_len == 0:
            raise ValueError("token_ids must not be empty.")

        pos0 = (kv_cache.seq_len if kv_cache is not None else 0) if start_pos is None else int(start_pos)
        if pos0 < 0:
            raise ValueError(f"start_pos must be >= 0, got {pos0}")
        if pos0 + seq_len > self.max_context_length:
            raise ContextLengthExceededError(
                f"Sequence length {pos0 + seq_len} (start_pos={pos0} + seq_len={seq_len}) "
                f"exceeds max_context_length={self.max_context_length}."
            )

        cfg = self.config
        hidden_states = self._lookup_embeddings(token_ids)  # (seq_len, embedding_length)

        position_ids = np.arange(pos0, pos0 + seq_len, dtype=np.int64)
        cos, sin = self._compute_rope_embeddings(position_ids)

        num_kv_groups = cfg.head_count // cfg.head_count_kv
        scaling = np.float32(1.0 / math.sqrt(cfg.key_length))

        for layer_idx in range(cfg.block_count):
            prefix = f"blk.{layer_idx}."
            residual = hidden_states

            attn_norm_w = self._get_tensor(f"{prefix}attn_norm.weight")
            normed = self._rms_norm(hidden_states, attn_norm_w, cfg.rms_norm_eps)

            q = self._matmul_weight(normed, f"{prefix}attn_q.weight", f"{prefix}attn_q.bias")
            k = self._matmul_weight(normed, f"{prefix}attn_k.weight", f"{prefix}attn_k.bias")
            v = self._matmul_weight(normed, f"{prefix}attn_v.weight", f"{prefix}attn_v.bias")

            # Reshape to (heads, seq_len, head_dim)
            q = q.reshape(seq_len, cfg.head_count, cfg.key_length).transpose(1, 0, 2)
            k = k.reshape(seq_len, cfg.head_count_kv, cfg.key_length).transpose(1, 0, 2)
            v = v.reshape(seq_len, cfg.head_count_kv, cfg.value_length).transpose(1, 0, 2)

            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

            if kv_cache is not None:
                k, v = kv_cache.update(layer_idx, k, v, start_pos=pos0)

            total_seq_len = k.shape[1]
            k_rep = self._repeat_kv(k, num_kv_groups)
            v_rep = self._repeat_kv(v, num_kv_groups)

            # Scaled dot-product attention: (head_count, seq_len, total_seq_len)
            attn_weights = np.matmul(q, np.swapaxes(k_rep, -1, -2)) * scaling

            if seq_len > 1 or total_seq_len > pos0 + seq_len:
                q_positions = (pos0 + np.arange(seq_len, dtype=np.int64))[:, None]
                k_positions = np.arange(total_seq_len, dtype=np.int64)[None, :]
                causal_mask = k_positions > q_positions
                attn_weights = np.where(causal_mask[None, :, :], np.float32(-np.inf), attn_weights)

            max_scores = np.max(attn_weights, axis=-1, keepdims=True)
            exp_scores = np.exp(attn_weights - max_scores)
            attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            attn_out = np.matmul(attn_probs, v_rep)  # (head_count, seq_len, value_length)
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, cfg.head_count * cfg.value_length)
            attn_proj = self._matmul_weight(attn_out, f"{prefix}attn_output.weight", f"{prefix}attn_output.bias")
            hidden_states = residual + attn_proj

            # Post-attention RMSNorm + SwiGLU FFN
            residual = hidden_states
            post_norm_name = (
                f"{prefix}post_attention_norm.weight"
                if f"{prefix}post_attention_norm.weight" in self.loader.tensors
                else f"{prefix}ffn_norm.weight"
            )
            post_norm_w = self._get_tensor(post_norm_name)
            ffn_normed = self._rms_norm(hidden_states, post_norm_w, cfg.rms_norm_eps)

            gate = self._matmul_weight(ffn_normed, f"{prefix}ffn_gate.weight")
            up = self._matmul_weight(ffn_normed, f"{prefix}ffn_up.weight")
            # SiLU(x) = x * sigmoid(x)
            silu_gate = gate / (np.float32(1.0) + np.exp(-np.clip(gate, -88.0, 88.0)))
            ffn_hidden = silu_gate * up
            ffn_out = self._matmul_weight(ffn_hidden, f"{prefix}ffn_down.weight")

            hidden_states = residual + ffn_out
            if layer_callback is not None:
                layer_callback(layer_idx, hidden_states)

        output_norm_w = self._get_tensor("output_norm.weight")
        hidden_states = self._rms_norm(hidden_states, output_norm_w, cfg.rms_norm_eps)

        output_w_name = "output.weight" if "output.weight" in self.loader.tensors else "token_embd.weight"
        target_h = hidden_states[-1:] if last_token_only else hidden_states
        logits = self._matmul_weight(target_h, output_w_name)
        if last_token_only:
            return logits[0]
        return logits

    def _resolve_input_tokens(self, prompt_or_tokens: Union[str, Sequence[int]]) -> List[int]:
        if isinstance(prompt_or_tokens, str):
            if len(prompt_or_tokens) == 0:
                raise ValueError("Prompt string must not be empty.")
            if self.tokenizer is None:
                raise ValueError("GGUF file does not contain tokenizer metadata to encode string prompts.")
            tokens = self.tokenizer.encode(prompt_or_tokens, add_bos=False, parse_special=True)
            if len(tokens) == 0:
                raise ValueError("Encoded prompt produced zero tokens.")
            return tokens

        if isinstance(prompt_or_tokens, (list, tuple, np.ndarray)):
            tokens = [int(t) for t in prompt_or_tokens]
            if len(tokens) == 0:
                raise ValueError("Prompt token sequence must not be empty.")
            return tokens

        raise TypeError(f"Unsupported prompt_or_tokens type: {type(prompt_or_tokens).__name__}")

    def generate_stream(
        self,
        prompt_or_tokens: Union[str, Sequence[int]],
        *,
        sampler_config: Optional[SamplerConfig] = None,
        max_new_tokens: int = 64,
        stop_token_ids: Optional[Iterable[int]] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cancel_callback: Optional[Callable[..., bool]] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Autoregressively stream generated tokens using real forward passes and KV caching.

        Yields dictionaries with keys:
        - `step`: int (0-indexed generation step)
        - `token_id`: int
        - `text`: str (decoded text delta for this token)
        - `accumulated_text`: str (generated text up to this step, trimmed of stop sequence)
        - `finish_reason`: Optional[str] (`None`, `"length"`, `"eos"`, `"stop_sequence"`, `"cancelled"`)
        - `logits`: np.ndarray (1D float32 logits that produced this token)
        """
        input_ids = self._resolve_input_tokens(prompt_or_tokens)
        if len(input_ids) > self.max_context_length:
            raise ContextLengthExceededError(
                f"Prompt token length {len(input_ids)} exceeds max_context_length={self.max_context_length}"
            )
        if max_new_tokens <= 0:
            return

        total_needed = min(self.max_context_length, len(input_ids) + max_new_tokens)
        cache = kv_cache if kv_cache is not None else self.create_kv_cache(max_seq_len=total_needed)
        sampler = Sampler(sampler_config or SamplerConfig(temperature=0.0, top_k=1))

        if stop_token_ids is None:
            stop_ids = {self.config.eos_token_id}
        else:
            stop_ids = {int(t) for t in stop_token_ids}

        stop_strs = [s for s in (stop_sequences or []) if s]

        all_tokens: List[int] = list(input_ids)
        generated_tokens: List[int] = []
        accumulated_text = ""

        self.active_sessions += 1
        try:
            # Prefill prompt into KV cache (accounting for any prefix already in cache)
            cached_prefix = cache.seq_len
            if cached_prefix > 0 and cached_prefix < len(input_ids):
                prefill_tokens = input_ids[cached_prefix:]
                start_pos = cached_prefix
            else:
                if cached_prefix > 0:
                    cache.reset()
                prefill_tokens = input_ids
                start_pos = 0

            logits_2d = self.forward(prefill_tokens, kv_cache=cache, start_pos=start_pos)
            next_logits = logits_2d[-1]

            for step in range(max_new_tokens):
                if cancel_callback is not None:
                    try:
                        should_continue = cancel_callback(step)
                    except TypeError:
                        should_continue = cancel_callback()
                    if should_continue is False:
                        return

                next_token = sampler.sample(next_logits, previous_tokens=all_tokens)
                generated_tokens.append(next_token)
                all_tokens.append(next_token)

                piece = (
                    self.tokenizer.decode([next_token], skip_special_tokens=False)
                    if self.tokenizer is not None
                    else ""
                )
                if self.tokenizer is not None:
                    accumulated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=False
                    )
                else:
                    accumulated_text += piece

                finish_reason: Optional[str] = None
                if next_token in stop_ids:
                    finish_reason = "eos"

                matched_stop_seq: Optional[str] = None
                for s_seq in stop_strs:
                    if s_seq in accumulated_text:
                        matched_stop_seq = s_seq
                        if finish_reason is None:
                            finish_reason = "stop_sequence"
                        break

                if matched_stop_seq is not None:
                    cut_idx = accumulated_text.find(matched_stop_seq)
                    trimmed_text = accumulated_text[:cut_idx]
                    yield {
                        "step": step,
                        "token_id": next_token,
                        "text": piece,
                        "accumulated_text": trimmed_text,
                        "stop_sequence": matched_stop_seq,
                        "finish_reason": finish_reason,
                        "logits": next_logits,
                    }
                    return

                if finish_reason is None and (
                    step + 1 >= max_new_tokens or len(all_tokens) >= self.max_context_length
                ):
                    finish_reason = "length"

                yield {
                    "step": step,
                    "token_id": next_token,
                    "text": piece,
                    "accumulated_text": accumulated_text,
                    "stop_sequence": None,
                    "finish_reason": finish_reason,
                    "logits": next_logits,
                }

                if finish_reason is not None:
                    return

                # Advance KV cache with the newly sampled token
                step_logits_2d = self.forward([next_token], kv_cache=cache, start_pos=cache.seq_len)
                next_logits = step_logits_2d[-1]
        finally:
            self.active_sessions = max(0, self.active_sessions - 1)
            if kv_cache is None:
                cache.reset()

    def generate(
        self,
        prompt_or_tokens: Union[str, Sequence[int]],
        *,
        sampler_config: Optional[SamplerConfig] = None,
        max_new_tokens: int = 64,
        stop_token_ids: Optional[Iterable[int]] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        cancel_callback: Optional[Callable[..., bool]] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Dict[str, Any]:
        """
        Run non-streaming generation and return generated token IDs, decoded text,
        first-step logits, and finish reason.
        """
        token_ids: List[int] = []
        first_logits: Optional[np.ndarray] = None
        final_text = ""
        finish_reason: Optional[str] = None
        matched_stop: Optional[str] = None

        for event in self.generate_stream(
            prompt_or_tokens,
            sampler_config=sampler_config,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            stop_sequences=stop_sequences,
            cancel_callback=cancel_callback,
            kv_cache=kv_cache,
        ):
            if first_logits is None:
                first_logits = event["logits"]
            token_ids.append(int(event["token_id"]))
            final_text = event["accumulated_text"]
            finish_reason = event["finish_reason"]
            if event.get("stop_sequence") is not None:
                matched_stop = event["stop_sequence"]

        if finish_reason is None:
            finish_reason = "cancelled" if cancel_callback is not None else "length"

        return {
            "token_ids": token_ids,
            "text": final_text,
            "first_logits": first_logits,
            "finish_reason": finish_reason,
            "stop_sequence": matched_stop,
        }

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
        kv_cache: Optional[KVCache] = None,
    ) -> Dict[str, Any]:
        """
        Format multi-turn chat messages via `Hermes36BAdapter`, run real token generation
        through `SeedOssEngine`, and parse `<think>` and `<tool_call>` tags into OpenAI
        `tool_calls` and Anthropic `tool_use` structures.
        """
        if not messages:
            raise ValueError("messages list must not be empty.")

        adapter = Hermes36BAdapter(thinking=thinking, keep_cots=keep_cots)
        formatted_prompt = adapter.format_chat_prompt(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )

        default_stops = ["<|eot_id|>", "<seed:eos>"]
        if stop_sequences:
            for s in stop_sequences:
                if s not in default_stops:
                    default_stops.append(s)

        effective_sampler = sampler_config or SamplerConfig(
            temperature=0.0,
            top_k=1,
            top_p=0.95,
        )

        gen_result = self.generate(
            formatted_prompt,
            sampler_config=effective_sampler,
            max_new_tokens=max_new_tokens,
            stop_sequences=default_stops,
            kv_cache=kv_cache,
        )

        raw_text = gen_result["text"]
        parsed = adapter.parse_generation(raw_text)

        return {
            "prompt": formatted_prompt,
            "token_ids": gen_result["token_ids"],
            "raw_output": raw_text,
            "content": parsed["final_content"],
            "thinking": parsed["thinking"],
            "has_thinking": parsed["has_thinking"],
            "unclosed_thinking": parsed["unclosed_thinking"],
            "tool_calls": parsed["tool_calls"],
            "has_tool_calls": parsed["has_tool_calls"],
            "malformed_tool_calls": parsed["malformed_tool_calls"],
            "openai_tool_calls": parsed["openai_tool_calls"],
            "anthropic_tool_use": parsed["anthropic_tool_use"],
            "finish_reason": gen_result["finish_reason"],
        }

    def close(self) -> None:
        if self._owns_loader:
            self.loader.close()

    def __enter__(self) -> "SeedOssEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


SeedOssForCausalLM = SeedOssEngine
