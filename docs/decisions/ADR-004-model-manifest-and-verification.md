# ADR-004: Model Manifest Schema, Verification, and Safety Gate

## Status
Accepted

## Context
Model files downloaded from the internet or imported from local storage represent untrusted binary inputs. Malformed headers, integer overflows, unexpected opcodes, or embedded scripts in serialization formats can crash the process or cause remote code execution (RCE).

## Decision
1. **Zero Remote Code Execution:** QERLIO will strictly prohibit loading arbitrary Python scripts, pickled weights, or unverified custom code during model loading.
2. **Deterministic Cryptographic Verification:** All imported models must have their SHA-256 checksum verified against an authorized manifest before registration into the active catalog.
3. **Strict Bounds Checking:** The GGUF header parser must enforce limits on:
   - Maximum tensor name lengths.
   - Maximum dimension counts and tensor counts.
   - Offset arithmetic to prevent integer overflow and buffer over-reads.
   - Absolute filesystem path traversal protections.
4. **Structured Manifest:** Every model is represented by a JSON manifest (`model_manifest.json`) adhering to schema version 1.0 specifying identifier, size, architecture, quantization, context hints, license SPDX, and compatibility flags.

## Consequences
- **Positive:** Immune to standard serialization exploits (e.g., pickle exploits); deterministic integrity guarantees; prevents path traversal and corrupted file crashes.
- **Negative:** Requires pre-computation of SHA-256 checksums which can introduce an initial import delay for multi-gigabyte models.
