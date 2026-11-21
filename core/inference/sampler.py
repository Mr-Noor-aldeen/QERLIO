"""
Native NumPy token sampler supporting greedy decoding, temperature scaling,
top-k filtering, top-p (nucleus) filtering, and repetition penalty.
Written from scratch using only the Python standard library and NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class SamplerConfig:
    """
    Sampling hyperparameters for causal language model generation.
    Defaults match official NousResearch/Hermes-4.3-36B generation parameters.
    """

    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    repeat_penalty: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.repeat_penalty <= 0.0:
            raise ValueError(f"repeat_penalty must be > 0.0, got {self.repeat_penalty}")


class Sampler:
    """
    Deterministic/stochastic token sampler implemented in pure NumPy.
    """

    def __init__(
        self,
        config: Optional[SamplerConfig] = None,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        base = config or SamplerConfig()
        self.config = SamplerConfig(
            temperature=base.temperature if temperature is None else float(temperature),
            top_k=base.top_k if top_k is None else int(top_k),
            top_p=base.top_p if top_p is None else float(top_p),
            repeat_penalty=base.repeat_penalty if repeat_penalty is None else float(repeat_penalty),
            seed=base.seed if seed is None else int(seed),
        )
        self._rng = np.random.Generator(np.random.PCG64(self.config.seed))

    @staticmethod
    def apply_repetition_penalty(
        logits: np.ndarray,
        previous_tokens: Optional[Iterable[int]],
        penalty: float,
    ) -> np.ndarray:
        """
        Apply CTRL / llama.cpp style repetition penalty to previously seen token IDs:
        if logit < 0 -> logit * penalty, else -> logit / penalty.
        """
        scores = np.array(logits, dtype=np.float64, copy=True)
        if previous_tokens is None or penalty == 1.0:
            return scores

        vocab_size = scores.shape[0]
        unique_tokens = {int(t) for t in previous_tokens if 0 <= int(t) < vocab_size}
        for tid in unique_tokens:
            val = scores[tid]
            scores[tid] = val * penalty if val < 0.0 else val / penalty
        return scores

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over 1D logits array."""
        max_logit = np.max(logits)
        if not np.isfinite(max_logit):
            raise ValueError("Cannot compute softmax over all -inf or NaN logits.")
        shifted = logits - max_logit
        exps = np.exp(shifted)
        total = np.sum(exps)
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("Invalid softmax normalization denominator.")
        return exps / total

    def sample_distribution(
        self,
        logits: np.ndarray,
        previous_tokens: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        """
        Compute the post-processed probability distribution over the vocabulary
        after repetition penalty, temperature scaling, top-k, and top-p filtering.
        """
        arr = np.asarray(logits)
        if arr.ndim == 2:
            arr = arr[-1]
        elif arr.ndim != 1:
            raise ValueError(f"Expected 1D or 2D logits array, got shape {arr.shape}")

        if arr.size == 0:
            raise ValueError("Cannot sample from empty logits array.")

        scores = self.apply_repetition_penalty(arr, previous_tokens, self.config.repeat_penalty)

        if self.config.temperature <= 0.0 or self.config.top_k == 1:
            probs = np.zeros_like(scores, dtype=np.float64)
            probs[int(np.argmax(scores))] = 1.0
            return probs

        scores = scores / self.config.temperature

        vocab_size = scores.shape[0]
        top_k = self.config.top_k
        if 0 < top_k < vocab_size:
            top_k_indices = np.argpartition(scores, -top_k)[-top_k:]
            mask = np.ones(vocab_size, dtype=bool)
            mask[top_k_indices] = False
            scores[mask] = -np.inf

        probs = self.softmax(scores)

        top_p = self.config.top_p
        if 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            # Mask tokens where cumulative probability BEFORE including the token exceeds top_p
            cutoff_mask = (cumulative_probs - sorted_probs) >= top_p
            cutoff_mask[0] = False  # Always retain the highest probability token
            sorted_probs[cutoff_mask] = 0.0
            probs = np.zeros_like(probs)
            probs[sorted_indices] = sorted_probs
            total = np.sum(probs)
            if total > 0.0:
                probs /= total

        return probs

    def sample(
        self,
        logits: np.ndarray,
        previous_tokens: Optional[Iterable[int]] = None,
    ) -> int:
        """
        Sample a single token ID from `logits` (shape `(vocab_size,)` or `(seq_len, vocab_size)`).
        """
        arr = np.asarray(logits)
        if arr.ndim == 2:
            arr = arr[-1]
        elif arr.ndim != 1:
            raise ValueError(f"Expected 1D or 2D logits array, got shape {arr.shape}")

        if arr.size == 0:
            raise ValueError("Cannot sample from empty logits array.")

        scores = self.apply_repetition_penalty(arr, previous_tokens, self.config.repeat_penalty)

        if self.config.temperature <= 0.0 or self.config.top_k == 1:
            return int(np.argmax(scores))

        probs = self.sample_distribution(arr, previous_tokens=previous_tokens)
        token_id = int(self._rng.choice(probs.shape[0], p=probs))
        return token_id
