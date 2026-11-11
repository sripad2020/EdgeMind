"""Deterministic GGUF re-quantization: F16 / BF16 / F32  ->  Q8_0 or Q4_0.

No learning involved: a fixed numerical transform that reproduces llama.cpp's reference
quantizers (``quantize_row_q8_0_ref`` / ``quantize_row_q4_0_ref``) with numpy, so the output
loads in llama.cpp / Ollama.

Rules
  * Only 2-D (or higher) weight tensors whose row length is a multiple of 32 and whose source
    type is F32/F16/BF16 are quantized; norms, biases and odd shapes are copied unchanged.
  * For a Q4_0 target, ``token_embd`` and ``output`` are written as Q8_0 ("mixed" precision),
    because they are the most quality-sensitive tensors - the same idea llama.cpp uses.
  * Metadata is copied byte-for-byte except ``general.file_type``.

Complexity: O(total parameters) time; memory O(largest chunk) because each tensor is
memory-mapped and processed in row chunks (~4M values at a time), never loaded whole.
The output is written to a temporary file and atomically renamed, so a crash never leaves a
half-written model behind.
"""

from __future__ import annotations

import os
import shutil
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Final

import numpy as np
import numpy.typing as npt

from ._validate import MiB
from .analysis.gguf import (
    DEFAULT_LIMITS,
    GGUF_MAGIC,
    SUPPORTED_VERSIONS,
    GGUFFile,
    GGUFValueType,
    TensorInfo,
    _read_value,
    _Reader,
    read_gguf,
)
from .analysis.profile import ModelProfile, build_profile
from .errors import AirError, ModelFormatError, ValidationError

QK: Final[int] = 32  # values per Q8_0 / Q4_0 block
_CHUNK_VALUES: Final[int] = 4 * 1024 * 1024
_SOURCE_DTYPES: Final[dict[int, str]] = {0: "F32", 1: "F16", 30: "BF16"}
_KEEP_HIGHER_FOR_Q4: Final[tuple[str, ...]] = ("token_embd.weight", "output.weight")

_Q8_DTYPE: Final = np.dtype([("d", "<f2"), ("qs", "i1", (QK,))])
_Q4_DTYPE: Final = np.dtype([("d", "<f2"), ("qs", "u1", (QK // 2,))])


class QuantTarget(str, Enum):
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"

    @property
    def ggml_type(self) -> int:
        return 8 if self is QuantTarget.Q8_0 else 2

    @property
    def file_type(self) -> int:  # llama.cpp LLAMA_FTYPE_MOSTLY_*
        return 7 if self is QuantTarget.Q8_0 else 2

    @property
    def bytes_per_value(self) -> float:
        return 34 / 32 if self is QuantTarget.Q8_0 else 18 / 32


class QuantizationError(AirError):
    pass


# ---- numerical kernels ---------------------------------------------------------------------


def _roundf(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """C ``roundf``: round half away from zero (numpy's round is half-to-even)."""
    return np.asarray(np.trunc(x + np.copysign(np.float32(0.5), x)), dtype=np.float32)


def quantize_q8_0(rows: npt.NDArray[np.float32]) -> bytes:
    """Q8_0: per 32 values, d = max|x| / 127 (fp16) and qs = roundf(x / d) (int8)."""
    blocks = rows.reshape(-1, QK)
    amax = np.abs(blocks).max(axis=1)
    d = amax / np.float32(127.0)
    inv = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d != 0)
    out = np.empty(blocks.shape[0], dtype=_Q8_DTYPE)
    out["d"] = d.astype(np.float16)
    out["qs"] = _roundf(blocks * inv[:, None]).astype(np.int8)
    return out.tobytes()


def quantize_q4_0(rows: npt.NDArray[np.float32]) -> bytes:
    """Q4_0: per 32 values, d = (value with max |x|) / -8 (fp16);
    q = min(15, trunc(x / d + 8.5)); byte j packs q[j] (low nibble) and q[j+16] (high)."""
    blocks = rows.reshape(-1, QK)
    idx = np.abs(blocks).argmax(axis=1)
    mx = blocks[np.arange(blocks.shape[0]), idx]
    d = mx / np.float32(-8.0)
    inv = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d != 0)
    q = np.clip(np.floor(blocks * inv[:, None] + np.float32(8.5)), 0, 15).astype(np.uint8)
    out = np.empty(blocks.shape[0], dtype=_Q4_DTYPE)
    out["d"] = d.astype(np.float16)
    out["qs"] = q[:, : QK // 2] | (q[:, QK // 2:] << 4)
    return out.tobytes()


def dequantize_q8_0(data: bytes) -> npt.NDArray[np.float32]:
    b = np.frombuffer(data, dtype=_Q8_DTYPE)
    values = b["qs"].astype(np.float32) * b["d"].astype(np.float32)[:, None]
    return np.asarray(values, dtype=np.float32).reshape(-1)


def dequantize_q4_0(data: bytes) -> npt.NDArray[np.float32]:
    b = np.frombuffer(data, dtype=_Q4_DTYPE)
    lo = (b["qs"] & 0x0F).astype(np.float32) - 8
    hi = (b["qs"] >> 4).astype(np.float32) - 8
    d = b["d"].astype(np.float32)[:, None]
    return np.asarray(np.concatenate([lo, hi], axis=1) * d, dtype=np.float32).reshape(-1)


# ---- planning -------------------------------------------------------------------------------


def _tensor_target(t: TensorInfo, target: QuantTarget) -> QuantTarget | None:
    """Which quantization a tensor gets, or None to copy it unchanged."""
    if t.ggml_type not in _SOURCE_DTYPES or len(t.shape) < 2 or t.shape[0] % QK:
        return None
    if target is QuantTarget.Q4_0 and t.name in _KEEP_HIGHER_FOR_Q4:
        return QuantTarget.Q8_0
    return target


def _quantized_bytes(t: TensorInfo, target: QuantTarget) -> int:
    tt = _tensor_target(t, target)
    return t.n_bytes if tt is None else int(t.n_elements * tt.bytes_per_value)


def estimate_output_bytes(gguf: GGUFFile, target: QuantTarget) -> int:
    return sum(_quantized_bytes(t, target) for t in gguf.tensors) + gguf.data_offset


def is_quantizable(profile: ModelProfile) -> bool:
    """Only full-precision sources can be quantized (re-quantizing Q4 would compound error)."""
    return profile.quant.name in ("F32", "F16", "BF16")


@dataclass(frozen=True, slots=True)
class QuantPlanItem:
    target: QuantTarget
    estimated_bytes: int
    quality_loss_prior: float


def plan_targets(profile: ModelProfile, max_quality_loss: float) -> list[QuantPlanItem]:
    """Deterministic rule: offer every target whose quality-loss prior is within the limit,
    highest precision first. Hardware fit is judged by the caller against live memory."""
    if not is_quantizable(profile):
        return []
    priors = {QuantTarget.Q8_0: 0.001, QuantTarget.Q4_0: 0.05}
    gguf = read_gguf(profile.path)
    return [
        QuantPlanItem(t, estimate_output_bytes(gguf, t), priors[t])
        for t in (QuantTarget.Q8_0, QuantTarget.Q4_0)
        if priors[t] <= max_quality_loss + 1e-9
    ]


# ---- GGUF writing ---------------------------------------------------------------------------


def _kv_spans(path: Path) -> tuple[int, list[tuple[str, int, int]], bytes]:
    """Return (version, [(key, start, end)], raw KV bytes) of a GGUF header. The KV section
    is re-parsed with the hardened reader so every length is bounds-checked."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        r = _Reader(fh, size)
        if r.read_exact(4) != GGUF_MAGIC:
            raise ModelFormatError("not a GGUF file")
        version = r.u32()
        if version not in SUPPORTED_VERSIONS:
            raise ModelFormatError(f"unsupported GGUF version {version}")
        r.u64()
        n_kv = r.u64()
        kv_start = r.pos
        spans: list[tuple[str, int, int]] = []
        for _ in range(n_kv):
            start = r.pos
            key = r.string(DEFAULT_LIMITS.max_key_bytes)
            _read_value(r, r.value_type(), DEFAULT_LIMITS)
            spans.append((key, start - kv_start, r.pos - kv_start))
        kv_end = r.pos
    with path.open("rb") as fh:
        fh.seek(kv_start)
        raw = fh.read(kv_end - kv_start)
    return version, spans, raw


def _encode_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _encode_u32_kv(key: str, value: int) -> bytes:
    return _encode_str(key) + struct.pack("<I", GGUFValueType.UINT32) + struct.pack("<I", value)


def _source_rows(src: Path, gguf: GGUFFile, t: TensorInfo) -> npt.NDArray[np.generic]:
    """Memory-map a tensor as a (rows, row_len) array in its source dtype (no copy)."""
    row_len = t.shape[0]
    n_rows = t.n_elements // row_len
    offset = gguf.data_offset + t.offset
    if t.ggml_type == 0:
        arr = np.memmap(src, dtype="<f4", mode="r", offset=offset, shape=(n_rows, row_len))
    elif t.ggml_type == 1:
        arr = np.memmap(src, dtype="<f2", mode="r", offset=offset, shape=(n_rows, row_len))
    else:  # BF16
        arr = np.memmap(src, dtype="<u2", mode="r", offset=offset, shape=(n_rows, row_len))
    return arr


def _to_f32(chunk: npt.NDArray[np.generic], ggml_type: int) -> npt.NDArray[np.float32]:
    if ggml_type == 30:  # BF16 -> F32: place the 16 bits in the high half of a float32
        return (np.asarray(chunk).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(chunk, dtype=np.float32)


def _write_tensor(
    out: BinaryIO, src: Path, gguf: GGUFFile, t: TensorInfo, tt: QuantTarget | None
) -> None:
    if tt is None:  # copy unchanged, in chunks
        with src.open("rb") as fh:
            fh.seek(gguf.data_offset + t.offset)
            remaining = t.n_bytes
            while remaining:
                buf = fh.read(min(remaining, 8 * MiB))
                if not buf:
                    raise ModelFormatError(f"unexpected end of data in {t.name}")
                out.write(buf)
                remaining -= len(buf)
        return
    rows = _source_rows(src, gguf, t)
    step = max(1, _CHUNK_VALUES // t.shape[0])
    kernel = quantize_q8_0 if tt is QuantTarget.Q8_0 else quantize_q4_0
    for start in range(0, rows.shape[0], step):
        out.write(kernel(_to_f32(rows[start:start + step], t.ggml_type)))
    del rows


_Plan = list[tuple[TensorInfo, "QuantTarget | None"]]


def _write_header(
    out: BinaryIO, *, version: int, kv_parts: list[bytes], plan: _Plan, offsets: list[int],
    align: int,
) -> None:
    out.write(GGUF_MAGIC)
    out.write(struct.pack("<IQQ", version, len(plan), len(kv_parts)))
    for part in kv_parts:
        out.write(part)
    for (t, tt), off in zip(plan, offsets, strict=True):
        out.write(_encode_str(t.name))
        out.write(struct.pack("<I", len(t.shape)))
        for dim in t.shape:
            out.write(struct.pack("<Q", dim))
        out.write(struct.pack("<IQ", t.ggml_type if tt is None else tt.ggml_type, off))
    out.write(b"\0" * ((-out.tell()) % align))


def _write_data(
    out: BinaryIO, *, src: Path, gguf: GGUFFile, plan: _Plan, offsets: list[int],
    progress: Callable[[int, int], None] | None,
) -> None:
    data_start = out.tell()
    for i, ((t, tt), off) in enumerate(zip(plan, offsets, strict=True)):
        pad = data_start + off - out.tell()
        if pad < 0:
            raise QuantizationError("internal layout error")
        out.write(b"\0" * pad)
        _write_tensor(out, src, gguf, t, tt)
        if progress is not None:
            progress(i + 1, len(plan))


def quantize_gguf(
    src: Path,
    dst: Path,
    target: QuantTarget,
    *,
    progress: Callable[[int, int], None] | None = None,
    override: Callable[[TensorInfo, QuantTarget], QuantTarget] | None = None,
) -> ModelProfile:
    """Write ``dst`` as a ``target``-quantized copy of ``src``. Returns the new profile.

    ``override(tensor, default)`` may pick a different precision per quantizable tensor
    (mixed precision, e.g. keep some layers at Q8_0 in an otherwise Q4_0 file). Tensors
    that cannot be quantized (norms, 1-D) are always copied unchanged."""
    if not isinstance(target, QuantTarget):
        raise ValidationError("target must be a QuantTarget")
    gguf = read_gguf(src)
    profile = build_profile(gguf)
    if not is_quantizable(profile):
        raise QuantizationError(
            f"{profile.quant.name} source cannot be re-quantized; need F16/BF16/F32"
        )
    need = estimate_output_bytes(gguf, target)
    free = shutil.disk_usage(dst.parent).free
    if need * 1.1 > free:
        raise QuantizationError(f"not enough disk space: need ~{need >> 20} MiB, "
                                f"have {free >> 20} MiB")

    version, spans, raw_kv = _kv_spans(src)
    kv_parts = [
        raw_kv[s:e] if key != "general.file_type" else _encode_u32_kv(key, target.file_type)
        for key, s, e in spans
    ]
    if not any(key == "general.file_type" for key, _s, _e in spans):
        kv_parts.append(_encode_u32_kv("general.file_type", target.file_type))

    align = gguf.alignment
    plan: _Plan = []
    for t in gguf.tensors:
        tt = _tensor_target(t, target)
        if tt is not None and override is not None:
            tt = override(t, tt)
            if not isinstance(tt, QuantTarget):
                raise ValidationError("override must return a QuantTarget")
        plan.append((t, tt))
    offsets: list[int] = []
    cursor = 0
    for t, tt in plan:
        offsets.append(cursor)
        size = t.n_bytes if tt is None else int(t.n_elements * tt.bytes_per_value)
        cursor = (cursor + size + align - 1) // align * align

    tmp = dst.with_name(dst.name + ".partial")
    try:
        with tmp.open("wb") as out:
            _write_header(out, version=version, kv_parts=kv_parts, plan=plan,
                          offsets=offsets, align=align)
            _write_data(out, src=src, gguf=gguf, plan=plan, offsets=offsets,
                        progress=progress)
        new_profile = build_profile(read_gguf(tmp))  # validate with the hardened parser
        if new_profile.family_key != profile.family_key:
            raise QuantizationError("quantized model does not match the source architecture")
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return build_profile(read_gguf(dst))


def derived_model_name(source: str, target: QuantTarget) -> str:
    """Name for a quantized copy: 'smollm2:360m' -> 'smollm2:360m-air-q8_0'."""
    return f"{source}-air-{target.value}" if ":" in source else f"{source}:air-{target.value}"


@dataclass(frozen=True, slots=True)
class QuantizeResult:
    model_name: str
    target: QuantTarget
    size_bytes: int
    seconds: float
    created: bool  # False if it already existed


def timed_quantize(src: Path, dst: Path, target: QuantTarget) -> tuple[ModelProfile, float]:
    t0 = time.perf_counter()
    profile = quantize_gguf(src, dst, target)
    return profile, time.perf_counter() - t0
