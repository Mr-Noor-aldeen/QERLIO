"""Native GGUF BPE Tokenizer implemented from scratch using only the Python standard library.

Supports GPT-2 / Seed-Coder byte-level BPE with NFC normalization, Unicode category
pre-tokenization (including individual digit splitting), and special/added token matching.
Zero runtime dependencies on `transformers`, `tokenizers`, `tiktoken`, `gguf`, or `torch`.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _bytes_to_unicode() -> Dict[int, str]:
    """Returns the standard GPT-2 reversible mapping from byte values (0..255) to Unicode characters."""
    bs: List[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs: List[int] = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _unicode_to_bytes() -> Dict[str, int]:
    """Returns the inverse GPT-2 mapping from Unicode characters to byte values (0..255)."""
    return {v: k for k, v in _bytes_to_unicode().items()}


_BYTE_ENCODER: Dict[int, str] = _bytes_to_unicode()
_BYTE_DECODER: Dict[str, int] = _unicode_to_bytes()


def _build_unicode_category_ranges() -> Tuple[str, str]:
    """Builds compact character class range strings for Unicode categories \\p{L} and \\p{N}."""
    l_cps: List[int] = []
    n_cps: List[int] = []
    for cp in range(sys.maxunicode + 1):
        cat = unicodedata.category(chr(cp))
        if cat.startswith("L"):
            l_cps.append(cp)
        elif cat.startswith("N"):
            n_cps.append(cp)

    def _compress_ranges(cps: List[int]) -> str:
        if not cps:
            return ""
        ranges: List[Tuple[int, int]] = []
        start = prev = cps[0]
        for cp in cps[1:]:
            if cp == prev + 1:
                prev = cp
            else:
                ranges.append((start, prev))
                start = prev = cp
        ranges.append((start, prev))

        parts: List[str] = []
        for s, e in ranges:
            sc = re.escape(chr(s))
            ec = re.escape(chr(e))
            if s == e:
                parts.append(sc)
            elif s + 1 == e:
                parts.append(sc + ec)
            else:
                parts.append(f"{sc}-{ec}")
        return "".join(parts)

    return _compress_ranges(l_cps), _compress_ranges(n_cps)


_UNICODE_L_RANGES, _UNICODE_N_RANGES = _build_unicode_category_ranges()

# Exact translation of NousResearch/Hermes-4.3-36B (`seed-coder`) pre-tokenizer Split regex:
# (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+
_SEED_CODER_PATTERN: re.Pattern[str] = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    rf"[^\r\n{_UNICODE_L_RANGES}{_UNICODE_N_RANGES}]?[{_UNICODE_L_RANGES}]+|"
    rf"[{_UNICODE_N_RANGES}]{{1}}|"
    rf" ?[^\s{_UNICODE_L_RANGES}{_UNICODE_N_RANGES}\r\n]+|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

# Classic GPT-2 pre-tokenizer regex (\p{N}+ instead of \p{N}{1})
_GPT2_PATTERN: re.Pattern[str] = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d|"
    rf" ?[{_UNICODE_L_RANGES}]+|"
    rf" ?[{_UNICODE_N_RANGES}]+|"
    rf" ?[^\s{_UNICODE_L_RANGES}{_UNICODE_N_RANGES}]+|"
    r"\s+(?!\S)|"
    r"\s+"
)


def _read_gguf_string(f: BinaryIO) -> str:
    raw_len = f.read(8)
    if len(raw_len) < 8:
        raise EOFError("Unexpected EOF reading GGUF string length")
    (n,) = struct.unpack("<Q", raw_len)
    data = f.read(n)
    if len(data) < n:
        raise EOFError("Unexpected EOF reading GGUF string payload")
    return data.decode("utf-8", errors="replace")


def _read_gguf_value(f: BinaryIO, vtype: int) -> Any:
    if vtype == 0:  # UINT8
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:  # INT8
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:  # UINT16
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:  # INT16
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:  # UINT32
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:  # INT32
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:  # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:  # BOOL
        return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == 8:  # STRING
        return _read_gguf_string(f)
    if vtype == 9:  # ARRAY
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type == 5:
            return list(struct.unpack(f"<{count}i", f.read(4 * count)))
        if elem_type == 4:
            return list(struct.unpack(f"<{count}I", f.read(4 * count)))
        if elem_type == 6:
            return list(struct.unpack(f"<{count}f", f.read(4 * count)))
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    if vtype == 10:  # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:  # INT64
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:  # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def read_gguf_tokenizer_metadata(path: str) -> Dict[str, Any]:
    """Reads GGUF header and extracts all metadata key-value pairs using only `struct`."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic {magic!r} in {path}")
        version, _tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        metadata: Dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_gguf_value(f, vtype)
        return metadata


class GGUFTokenizer:
    """Pure-Python Byte-Level BPE Tokenizer constructed directly from GGUF tokenizer tables."""

    # GGUF token type constants
    TOKEN_TYPE_NORMAL = 1
    TOKEN_TYPE_UNKNOWN = 2
    TOKEN_TYPE_CONTROL = 3
    TOKEN_TYPE_USER_DEFINED = 4
    TOKEN_TYPE_UNUSED = 5
    TOKEN_TYPE_BYTE = 6

    def __init__(
        self,
        tokens: Sequence[str],
        merges: Sequence[Union[str, Tuple[str, str]]],
        token_types: Optional[Sequence[int]] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 155127,
        pad_token_id: int = 1,
        pre_type: str = "seed-coder",
        normalize_nfc: bool = True,
        special_tokens: Optional[Dict[str, int]] = None,
    ) -> None:
        self.id_to_token: List[str] = list(tokens)
        self.vocab_size: int = len(self.id_to_token)
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.id_to_token)}

        self.token_types: List[int] = (
            list(token_types) if token_types is not None else [self.TOKEN_TYPE_NORMAL] * self.vocab_size
        )
        self.bos_token_id: int = int(bos_token_id)
        self.eos_token_id: int = int(eos_token_id)
        self.pad_token_id: int = int(pad_token_id)
        self.pre_type: str = pre_type
        self.normalize_nfc: bool = normalize_nfc

        # Build merge_ranks: dict[tuple[str, str], int]
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        for rank, entry in enumerate(merges):
            if isinstance(entry, tuple):
                pair = (entry[0], entry[1])
            else:
                parts = entry.split(" ", 1)
                if len(parts) != 2:
                    continue
                pair = (parts[0], parts[1])
            if pair not in self.merge_ranks:
                self.merge_ranks[pair] = rank

        # Identify added/special tokens (GGUF token_type 3 = CONTROL, 4 = USER_DEFINED)
        # Note: both types 3 and 4 are matched literally before pre-tokenization in HF tokenizers.
        # Only type 3 (CONTROL) tokens have `special=True` for `skip_special_tokens=True` in decode.
        self.added_token_to_id: Dict[str, int] = {}
        self.control_token_ids: set[int] = set()
        self.added_token_ids: set[int] = set()

        for idx, (tok, ttype) in enumerate(zip(self.id_to_token, self.token_types)):
            if ttype in (self.TOKEN_TYPE_CONTROL, self.TOKEN_TYPE_USER_DEFINED):
                self.added_token_to_id[tok] = idx
                self.added_token_ids.add(idx)
                if ttype == self.TOKEN_TYPE_CONTROL:
                    self.control_token_ids.add(idx)

        if special_tokens:
            for tok, idx in special_tokens.items():
                self.added_token_to_id[tok] = idx
                self.added_token_ids.add(idx)
                self.control_token_ids.add(idx)

        # Longest-match regex for special/added tokens
        if self.added_token_to_id:
            sorted_added = sorted(self.added_token_to_id.keys(), key=len, reverse=True)
            self._special_regex: Optional[re.Pattern[str]] = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_added) + ")"
            )
        else:
            self._special_regex = None

        if self.pre_type in ("seed-coder", "seed_oss", "llama3", "qwen2", "qwen3", "qwen35"):
            self._pre_pattern: re.Pattern[str] = _SEED_CODER_PATTERN
        else:
            self._pre_pattern = _GPT2_PATTERN

        self._bpe_cache: Dict[str, List[int]] = {}

    @classmethod
    def from_gguf_metadata(cls, metadata: Dict[str, Any]) -> "GGUFTokenizer":
        """Constructs a GGUFTokenizer from a parsed GGUF header metadata dictionary."""
        tokens = metadata.get("tokenizer.ggml.tokens")
        merges = metadata.get("tokenizer.ggml.merges")
        if tokens is None or merges is None:
            raise KeyError("GGUF metadata must contain 'tokenizer.ggml.tokens' and 'tokenizer.ggml.merges'")
        token_types = metadata.get("tokenizer.ggml.token_type")
        bos_id = metadata.get("tokenizer.ggml.bos_token_id", 0)
        eos_id = metadata.get("tokenizer.ggml.eos_token_id", 155127)
        pad_id = metadata.get("tokenizer.ggml.padding_token_id", 1)
        pre_type = metadata.get("tokenizer.ggml.pre", "seed-coder")
        return cls(
            tokens=tokens,
            merges=merges,
            token_types=token_types,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            pre_type=pre_type,
        )

    @classmethod
    def from_gguf_file(cls, path: str) -> "GGUFTokenizer":
        """Constructs a GGUFTokenizer directly from a `.gguf` file or GGUF header prefix binary."""
        metadata = read_gguf_tokenizer_metadata(path)
        return cls.from_gguf_metadata(metadata)

    def _bpe(self, token_mapped: str) -> List[int]:
        """Applies standard BPE merge loop to a byte-level-mapped pre-token string."""
        cached = self._bpe_cache.get(token_mapped)
        if cached is not None:
            return cached

        word: List[str] = list(token_mapped)
        if not word:
            return []

        while len(word) > 1:
            best_rank: Optional[int] = None
            best_pair: Optional[Tuple[str, str]] = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            merged_piece = first + second
            new_word: List[str] = []
            i = 0
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(merged_piece)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        ids: List[int] = []
        for piece in word:
            tid = self.token_to_id.get(piece)
            if tid is not None:
                ids.append(tid)
            else:
                # Fallback to individual byte-level chars if an unmerged sequence occurred
                for ch in piece:
                    ids.append(self.token_to_id[ch])

        self._bpe_cache[token_mapped] = ids
        return ids

    def _encode_chunk(self, text: str) -> List[int]:
        """Normalizes, pre-tokenizes, byte-maps, and BPE-encodes a non-special text chunk."""
        if not text:
            return []
        if self.normalize_nfc:
            text = unicodedata.normalize("NFC", text)
        token_ids: List[int] = []
        for match in self._pre_pattern.finditer(text):
            piece = match.group(0)
            raw_bytes = piece.encode("utf-8")
            mapped = "".join(_BYTE_ENCODER[b] for b in raw_bytes)
            token_ids.extend(self._bpe(mapped))
        return token_ids

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        parse_special: bool = True,
    ) -> List[int]:
        """Encodes a UTF-8 string into a list of integer token IDs."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        result: List[int] = []
        if add_bos:
            result.append(self.bos_token_id)

        if not text:
            return result

        if parse_special and self._special_regex is not None:
            parts = self._special_regex.split(text)
            for part in parts:
                if not part:
                    continue
                special_id = self.added_token_to_id.get(part)
                if special_id is not None:
                    result.append(special_id)
                else:
                    result.extend(self._encode_chunk(part))
        else:
            result.extend(self._encode_chunk(text))

        return result

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decodes a sequence of integer token IDs back into a UTF-8 string."""
        output_parts: List[str] = []
        byte_buffer = bytearray()

        def _flush_bytes() -> None:
            if byte_buffer:
                output_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for tid in token_ids:
            if tid < 0 or tid >= self.vocab_size:
                raise ValueError(f"Token ID {tid} out of vocabulary range [0, {self.vocab_size})")

            if skip_special_tokens and tid in self.control_token_ids:
                continue

            tok_str = self.id_to_token[tid]
            if tid in self.added_token_ids or self.token_types[tid] == self.TOKEN_TYPE_UNUSED:
                _flush_bytes()
                output_parts.append(tok_str)
            else:
                # Normal BPE token: map GPT-2 unicode chars back to raw bytes
                # Some unmerged placeholder tokens in vocab (e.g., IDs 7..13) are literal ASCII strings
                if all(ch in _BYTE_DECODER for ch in tok_str):
                    for ch in tok_str:
                        byte_buffer.append(_BYTE_DECODER[ch])
                else:
                    _flush_bytes()
                    output_parts.append(tok_str)

        _flush_bytes()
        return "".join(output_parts)
