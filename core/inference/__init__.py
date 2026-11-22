"""Native GGUF v3 inference engine package (from-scratch NumPy + stdlib implementation)."""

from pathlib import Path
from typing import Optional, Union

from core.inference.dequant import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_INFO,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_NAMES,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    compute_tensor_bytes,
    dequantize_block_scalar_reference,
    dequantize_bytes,
    get_block_size,
    get_type_size,
)
from core.inference.gguf_loader import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_SUPPORTED_VERSIONS,
    GGUFLoader,
    GGUFTensorInfo,
)
from core.inference.kv_cache import (
    ContextLengthExceededError,
    InsufficientMemoryError,
    KVCache,
    get_available_memory_bytes,
)
from core.inference.qwen35_model import (
    Qwen35Config,
    Qwen35Engine,
    Qwen35ForCausalLM,
    Qwen35HybridCache,
)
from core.inference.sampler import (
    Sampler,
    SamplerConfig,
)
from core.inference.seed_oss_model import (
    SeedOssConfig,
    SeedOssEngine,
    SeedOssForCausalLM,
    UnsupportedArchitectureError,
)
from core.inference.tokenizer import (
    GGUFTokenizer,
    read_gguf_tokenizer_metadata,
)


def load_gguf_engine(
    gguf_path: Union[str, Path],
    *,
    max_context_length: Optional[int] = None,
    preload_weights: Optional[bool] = None,
    memory_limit_bytes: Optional[int] = None,
) -> Union[SeedOssEngine, Qwen35Engine]:
    """Load the appropriate native inference engine (`SeedOssEngine` or `Qwen35Engine`)
    based on `general.architecture` in the GGUF header.
    """
    with GGUFLoader(gguf_path, header_only=True) as hdr:
        arch = hdr.metadata.get("general.architecture")

    if arch == "seed_oss":
        return SeedOssEngine(
            gguf_path,
            max_context_length=max_context_length,
            preload_weights=True if preload_weights is None else preload_weights,
            memory_limit_bytes=memory_limit_bytes,
        )
    if arch == "qwen35":
        return Qwen35Engine(
            gguf_path,
            max_context_length=max_context_length,
            preload_weights=preload_weights,
            memory_limit_bytes=memory_limit_bytes,
        )
    raise UnsupportedArchitectureError(
        f"Unsupported GGUF architecture: {arch!r}. Supported: 'seed_oss', 'qwen35'."
    )


__all__ = [
    "ContextLengthExceededError",
    "GGML_TYPE_BF16",
    "GGML_TYPE_F16",
    "GGML_TYPE_F32",
    "GGML_TYPE_INFO",
    "GGML_TYPE_IQ3_S",
    "GGML_TYPE_IQ4_NL",
    "GGML_TYPE_IQ4_XS",
    "GGML_TYPE_NAMES",
    "GGML_TYPE_Q2_K",
    "GGML_TYPE_Q3_K",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_Q4_K",
    "GGML_TYPE_Q5_0",
    "GGML_TYPE_Q5_1",
    "GGML_TYPE_Q5_K",
    "GGML_TYPE_Q6_K",
    "GGML_TYPE_Q8_0",
    "GGUF_DEFAULT_ALIGNMENT",
    "GGUF_MAGIC",
    "GGUF_SUPPORTED_VERSIONS",
    "GGUFLoader",
    "GGUFTensorInfo",
    "GGUFTokenizer",
    "InsufficientMemoryError",
    "KVCache",
    "Qwen35Config",
    "Qwen35Engine",
    "Qwen35ForCausalLM",
    "Qwen35HybridCache",
    "Sampler",
    "SamplerConfig",
    "SeedOssConfig",
    "SeedOssEngine",
    "SeedOssForCausalLM",
    "UnsupportedArchitectureError",
    "compute_tensor_bytes",
    "dequantize_block_scalar_reference",
    "dequantize_bytes",
    "get_available_memory_bytes",
    "get_block_size",
    "get_type_size",
    "load_gguf_engine",
    "read_gguf_tokenizer_metadata",
]
