"""From-scratch C SIMD quantized GEMM/GEMV backend and sequential tensor streamer.

Uses only Python standard library (`ctypes`, `concurrent.futures`, `os`, `pathlib`,
`platform`, `shutil`, `subprocess`) and `numpy` to accelerate `Q4_K`, `Q6_K`, `Q8_0`,
and `F32` matrix multiplications implemented in `core/inference/native_gemm.c`.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.inference.dequant import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    compute_tensor_bytes,
    get_block_size,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = Path(__file__).resolve().with_name("native_gemm.c")
_INC_DIR = _REPO_ROOT / "include"
_BUILD_DIR = _REPO_ROOT / "build"

SUPPORTED_NATIVE_GEMM_TYPES = frozenset(
    {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F32}
)


def _find_llvm_tools_windows() -> Optional[Tuple[Path, Path]]:
    clang_exe = shutil.which("clang")
    lld_exe = shutil.which("lld-link")
    if clang_exe and lld_exe:
        return Path(clang_exe), Path(lld_exe)
    llvm_bin = Path("C:/Program Files/LLVM/bin")
    c_cand = llvm_bin / "clang.exe"
    l_cand = llvm_bin / "lld-link.exe"
    if c_cand.is_file() and l_cand.is_file():
        return c_cand, l_cand
    return None


def _compile_native_gemm_lib(target_path: Path) -> Optional[Path]:
    if not _SRC_PATH.is_file():
        return None
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sys_name = platform.system()
    machine = platform.machine().lower()

    pid_tag = os.getpid()
    tmp_target = target_path.with_name(f"{target_path.stem}.{pid_tag}{target_path.suffix}")
    obj_path = target_path.with_name(f"{target_path.stem}.{pid_tag}.obj")

    if sys_name == "Windows":
        tools = _find_llvm_tools_windows()
        if tools is None:
            return None
        clang_exe, lld_exe = tools
        arch_flags = ["-mavx2", "-mfma", "-mf16c"] if machine in ("amd64", "x86_64") else []
        res_c = subprocess.run(
            [
                str(clang_exe),
                "-c",
                "-O3",
                "-std=c11",
                "-DQERLIO_FREESTANDING",
                f"-I{_INC_DIR}",
                "-ffreestanding",
                "-fno-stack-protector",
                "-mno-stack-arg-probe",
                *arch_flags,
                str(_SRC_PATH),
                "-o",
                str(obj_path),
            ],
            capture_output=True,
            text=True,
        )
        if res_c.returncode != 0:
            return None
        res_l = subprocess.run(
            [str(lld_exe), "/DLL", "/NOENTRY", f"/OUT:{tmp_target}", str(obj_path)],
            capture_output=True,
            text=True,
        )
        obj_path.unlink(missing_ok=True)
        if res_l.returncode != 0 or not tmp_target.is_file():
            tmp_target.unlink(missing_ok=True)
            return None
        try:
            os.replace(tmp_target, target_path)
        except OSError:
            if target_path.is_file():
                tmp_target.unlink(missing_ok=True)
                return target_path
            return tmp_target
        return target_path

    # POSIX (Linux / macOS)
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    arch_flags = ["-march=native"] if machine in ("amd64", "x86_64") else []
    res = subprocess.run(
        [
            cc,
            "-shared",
            "-fPIC",
            "-O3",
            "-std=c11",
            "-DQERLIO_BUILD_SHARED",
            f"-I{_INC_DIR}",
            *arch_flags,
            str(_SRC_PATH),
            "-o",
            str(tmp_target),
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        # Retry without -march=native in case the VM/assembler restricts host flags
        res2 = subprocess.run(
            [
                cc,
                "-shared",
                "-fPIC",
                "-O3",
                "-std=c11",
                "-DQERLIO_BUILD_SHARED",
                f"-I{_INC_DIR}",
                str(_SRC_PATH),
                "-o",
                str(tmp_target),
            ],
            capture_output=True,
            text=True,
        )
        if res2.returncode != 0 or not tmp_target.is_file():
            tmp_target.unlink(missing_ok=True)
            return None
    os.replace(tmp_target, target_path)
    return target_path


def _try_load_gemm_lib(cand: Path) -> bool:
    if not cand.is_file():
        return False
    try:
        lib = ctypes.CDLL(str(cand))
        return hasattr(lib, "qerlio_gemm_q4_k_f32")
    except OSError:
        return False


def _discover_or_build_library() -> Optional[Path]:
    env_lib = os.environ.get("QERLIO_NATIVE_GEMM_LIB")
    if env_lib:
        env_path = Path(env_lib)
        if _try_load_gemm_lib(env_path):
            return env_path

    sys_name = platform.system()
    if sys_name == "Windows":
        ext = ".dll"
        cand_names = (
            f"native_gemm_avx2{ext}",
            f"native_gemm_runtime{ext}",
            f"native_gemm_scalar{ext}",
            "qerlio.dll",
            "libqerlio.dll",
        )
    elif sys_name == "Darwin":
        ext = ".dylib"
        cand_names = (f"native_gemm_runtime{ext}", "libqerlio.dylib")
    else:
        ext = ".so"
        cand_names = (f"native_gemm_runtime{ext}", "libqerlio.so")

    src_mtime = _SRC_PATH.stat().st_mtime if _SRC_PATH.is_file() else 0.0
    for base in (_BUILD_DIR, _BUILD_DIR / "Release", _BUILD_DIR / "Debug"):
        for cname in cand_names:
            cpath = base / cname
            if cpath.is_file() and cpath.stat().st_mtime >= src_mtime:
                if _try_load_gemm_lib(cpath):
                    return cpath

    compiled_cand = _BUILD_DIR / f"native_gemm_runtime{ext}"
    built = _compile_native_gemm_lib(compiled_cand)
    if built is not None and _try_load_gemm_lib(built):
        return built
    return None


class NativeGemmBackend:
    """Multi-threaded ctypes bridge to QERLIO's from-scratch C quantized GEMM kernels."""

    _singleton: Optional["NativeGemmBackend"] = None
    _singleton_checked: bool = False

    def __init__(self, lib_path: Path, max_workers: Optional[int] = None) -> None:
        self.lib_path = Path(lib_path)
        self._lib = ctypes.CDLL(str(self.lib_path))

        argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._fn_has_avx2 = self._lib.qerlio_gemm_has_avx2
        self._fn_has_avx2.restype = ctypes.c_int
        self._fn_has_avx2.argtypes = []

        self._kernels: Dict[int, ctypes._CFuncPtr] = {}
        for ggml_type, sym_name in (
            (GGML_TYPE_Q4_K, "qerlio_gemm_q4_k_f32"),
            (GGML_TYPE_Q6_K, "qerlio_gemm_q6_k_f32"),
            (GGML_TYPE_Q8_0, "qerlio_gemm_q8_0_f32"),
            (GGML_TYPE_F32, "qerlio_gemm_f32"),
        ):
            fn = getattr(self._lib, sym_name)
            fn.restype = ctypes.c_int
            fn.argtypes = argtypes
            self._kernels[ggml_type] = fn

        self.has_avx2: bool = bool(self._fn_has_avx2())
        cpu_n = os.cpu_count() or 4
        self.num_threads: int = (
            max(1, int(max_workers))
            if max_workers is not None
            else min(cpu_n, 24)
        )
        self._pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self.num_threads, thread_name_prefix="qerlio-gemm")
            if self.num_threads > 1
            else None
        )

    @classmethod
    def get_default(cls) -> Optional["NativeGemmBackend"]:
        if not cls._singleton_checked:
            cls._singleton_checked = True
            lib_path = _discover_or_build_library()
            if lib_path is not None:
                try:
                    cls._singleton = cls(lib_path)
                except OSError:
                    cls._singleton = None
        return cls._singleton

    @classmethod
    def probe_status(cls) -> str:
        inst = cls.get_default()
        if inst is None:
            return "Unavailable (using NumPy fallback)"
        simd = "AVX2+FMA+F16C" if inst.has_avx2 else "Scalar C"
        return f"Available ({simd}, {inst.num_threads} threads, {inst.lib_path.name})"

    def supports(self, ggml_type: int, in_dim: int) -> bool:
        if ggml_type not in self._kernels:
            return False
        block_size = get_block_size(ggml_type)
        return in_dim > 0 and (in_dim % block_size) == 0

    def matmul_raw_chunk(
        self,
        x: np.ndarray,
        w_chunk_ptr: int,
        y_out: np.ndarray,
        ggml_type: int,
        n_total: int,
        k: int,
        chunk_row_start: int,
        chunk_row_end: int,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute `y_out[..., chunk_row_start:chunk_row_end] = x @ W_chunk.T`."""
        fn = self._kernels[ggml_type]
        x2d = np.ascontiguousarray(x.reshape(-1, k), dtype=np.float32)
        m = int(x2d.shape[0])
        y2d = np.ascontiguousarray(y_out.reshape(m, n_total), dtype=np.float32)
        if y2d.ctypes.data != y_out.ctypes.data:
            raise ValueError("y_out must be contiguous float32")

        local_rows = int(chunk_row_end - chunk_row_start)
        row_bytes = compute_tensor_bytes(ggml_type, k)
        x_ptr = int(x2d.ctypes.data)
        y_ptr = int(y2d.ctypes.data)

        total_bytes = local_rows * row_bytes
        if self._pool is None or local_rows < 256 or total_bytes < (512 * 1024):
            rc = int(fn(x_ptr, w_chunk_ptr, y_ptr, m, n_total, k, chunk_row_start, chunk_row_end))
            if rc != 0:
                raise RuntimeError(f"native_gemm kernel returned error code {rc}")
        else:
            n_parts = min(self.num_threads, max(1, local_rows // 64))
            if n_parts <= 1:
                rc = int(
                    fn(x_ptr, w_chunk_ptr, y_ptr, m, n_total, k, chunk_row_start, chunk_row_end)
                )
                if rc != 0:
                    raise RuntimeError(f"native_gemm kernel returned error code {rc}")
            else:
                step = (local_rows + n_parts - 1) // n_parts
                ranges: List[Tuple[int, int]] = []
                cur = 0
                while cur < local_rows:
                    nxt = min(cur + step, local_rows)
                    ranges.append((cur, nxt))
                    cur = nxt

                futures: List[Future[int]] = []
                for rel_s, rel_e in ranges[:-1]:
                    abs_s = chunk_row_start + rel_s
                    abs_e = chunk_row_start + rel_e
                    ptr_s = w_chunk_ptr + rel_s * row_bytes
                    futures.append(self._pool.submit(fn, x_ptr, ptr_s, y_ptr, m, n_total, k, abs_s, abs_e))

                last_s, last_e = ranges[-1]
                rc_last = int(
                    fn(
                        x_ptr,
                        w_chunk_ptr + last_s * row_bytes,
                        y_ptr,
                        m,
                        n_total,
                        k,
                        chunk_row_start + last_s,
                        chunk_row_start + last_e,
                    )
                )
                if rc_last != 0:
                    raise RuntimeError(f"native_gemm kernel returned error code {rc_last}")

                for fut in futures:
                    rc_f = int(fut.result())
                    if rc_f != 0:
                        raise RuntimeError(f"native_gemm worker kernel returned error code {rc_f}")

        if bias is not None and chunk_row_end == n_total and chunk_row_start == 0:
            y_out += bias
        elif bias is not None and chunk_row_end > chunk_row_start:
            y_out[..., chunk_row_start:chunk_row_end] += bias[chunk_row_start:chunk_row_end]

        return y_out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


class SequentialTensorStreamer:
    """Double-buffered async sequential file reader bypassing OS mmap page-fault overhead.

    Maintains two reusable bytearrays and a dedicated 1-thread I/O executor so
    the next weight tensor slice can be pre-read while C GEMM computes on the current
    weight buffer.
    """

    def __init__(self, file_path: Path, initial_buf_bytes: int = 8 * 1024 * 1024) -> None:
        self.file_path = Path(file_path)
        self._fp = open(self.file_path, "rb", buffering=0)
        self._bufs = [bytearray(initial_buf_bytes), bytearray(initial_buf_bytes)]
        self._io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qerlio-io")
        self._active_slot: int = 0
        self._pending: Optional[Tuple[int, int, int, Future[int]]] = None

    def _read_exact_into_slot(self, slot: int, file_offset: int, n_bytes: int) -> int:
        if len(self._bufs[slot]) < n_bytes:
            self._bufs[slot] = bytearray(n_bytes)
        self._fp.seek(file_offset)
        mv = memoryview(self._bufs[slot])
        total = 0
        try:
            while total < n_bytes:
                n = self._fp.readinto(mv[total:n_bytes])
                if not n:
                    raise EOFError(
                        f"Unexpected EOF streaming {self.file_path}: expected {n_bytes} bytes at {file_offset}, got {total}"
                    )
                total += n
        finally:
            mv.release()
        return slot

    def _drain_pending(self) -> None:
        if self._pending is not None:
            _, _, _, fut = self._pending
            self._pending = None
            fut.result()

    def acquire(
        self,
        file_offset: int,
        n_bytes: int,
        prefetch: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Acquire a `uint8` NumPy view and raw pointer address for `[file_offset, file_offset + n_bytes)`.

        Wait-free if the requested slice was already prefetched into the alternate buffer.
        Optionally schedules an asynchronous prefetch of `prefetch=(next_offset, next_bytes)`
        into the other buffer before returning.
        """
        if n_bytes < 0:
            raise ValueError(f"n_bytes must be >= 0, got {n_bytes}")

        chosen_slot: int
        if self._pending is not None:
            p_slot, p_off, p_len, fut = self._pending
            self._pending = None
            fut.result()
            if p_off == file_offset and p_len == n_bytes:
                chosen_slot = p_slot
            else:
                chosen_slot = 1 - p_slot
                self._read_exact_into_slot(chosen_slot, file_offset, n_bytes)
        else:
            chosen_slot = self._active_slot
            self._read_exact_into_slot(chosen_slot, file_offset, n_bytes)

        self._active_slot = chosen_slot

        if prefetch is not None:
            next_off, next_len = prefetch
            if next_len > 0:
                next_slot = 1 - chosen_slot
                fut = self._io_pool.submit(
                    self._read_exact_into_slot, next_slot, next_off, next_len
                )
                self._pending = (next_slot, next_off, next_len, fut)

        arr_u8 = np.frombuffer(self._bufs[chosen_slot], dtype=np.uint8, count=n_bytes)
        return arr_u8, int(arr_u8.ctypes.data)

    def close(self) -> None:
        try:
            self._drain_pending()
        finally:
            self._io_pool.shutdown(wait=True)
            if not self._fp.closed:
                self._fp.close()
