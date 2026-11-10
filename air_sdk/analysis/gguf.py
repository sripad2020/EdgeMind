"""Hardened, streaming GGUF header parser (metadata + tensor index only).

Why a custom parser instead of handing the file straight to llama.cpp?
  * Defence in depth: malformed GGUF headers have historically triggered memory-safety
    bugs in native parsers. We validate the header in memory-safe Python first.
  * We only need metadata: tensor *data* is never read, so parsing a 40 GB model costs
    the same as parsing a 40 MB one.

Complexity
  Time:  O(K + T log T + S) where K = metadata entries, T = tensors, S = elements of
         string arrays that must be skipped (their lengths are variable).
         Fixed-width arrays are skipped with one seek, O(1).
  Space: O(K_small + T). Large arrays (e.g. a 150k-token vocabulary) are *summarised*
         (type + length), never materialised.

Security limits (``ParserLimits``)
  Every count/length read from the file is bounds-checked against both a hard cap and the
  number of bytes actually remaining in the file *before* any allocation or loop, so a
  hostile header cannot cause huge allocations or unbounded loops.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Final, TypeAlias

from .._validate import MiB
from ..errors import ModelFormatError

GGUF_MAGIC: Final[bytes] = b"GGUF"
GGUF_MAGIC_BIG_ENDIAN: Final[bytes] = b"FUGG"
SUPPORTED_VERSIONS: Final[frozenset[int]] = frozenset({2, 3})
DEFAULT_ALIGNMENT: Final[int] = 32
GGML_MAX_DIMS: Final[int] = 4


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_STRUCTS: Final[dict[GGUFValueType, struct.Struct]] = {
    GGUFValueType.UINT8: struct.Struct("<B"),
    GGUFValueType.INT8: struct.Struct("<b"),
    GGUFValueType.UINT16: struct.Struct("<H"),
    GGUFValueType.INT16: struct.Struct("<h"),
    GGUFValueType.UINT32: struct.Struct("<I"),
    GGUFValueType.INT32: struct.Struct("<i"),
    GGUFValueType.FLOAT32: struct.Struct("<f"),
    GGUFValueType.BOOL: struct.Struct("<B"),  # validated to 0/1 explicitly
    GGUFValueType.UINT64: struct.Struct("<Q"),
    GGUFValueType.INT64: struct.Struct("<q"),
    GGUFValueType.FLOAT64: struct.Struct("<d"),
}
_U32: Final = struct.Struct("<I")
_U64: Final = struct.Struct("<Q")

# Minimum encoded size of one element, used for "can the remaining file even hold this?"
# checks. Strings are at least their 8-byte length prefix; arrays at least type + count.
_MIN_ELEMENT_BYTES: Final[dict[GGUFValueType, int]] = {
    **{t: s.size for t, s in _SCALAR_STRUCTS.items()},
    GGUFValueType.STRING: 8,
    GGUFValueType.ARRAY: 12,
}

# ggml tensor types whose storage layout is well established: type id -> (block, bytes/block).
# Used to cross-check tensor extents. Other types fall back to offset spans.
GGML_TYPE_LAYOUT: Final[dict[int, tuple[int, int]]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    30: (1, 2),  # BF16
}


@dataclass(frozen=True, slots=True)
class ParserLimits:
    """Hard caps applied while parsing untrusted GGUF headers."""

    max_kv_count: int = 65_536
    max_tensor_count: int = 1 << 20
    max_key_bytes: int = 4_096
    max_string_bytes: int = 16 * MiB
    max_inline_array: int = 4_096
    max_array_depth: int = 2
    max_dim: int = 1 << 40
    max_tensor_elements: int = 1 << 50


DEFAULT_LIMITS: Final[ParserLimits] = ParserLimits()


@dataclass(frozen=True, slots=True)
class ArraySummary:
    """Placeholder for an array too large to materialise (only its shape is kept)."""

    element_type: GGUFValueType
    length: int


Scalar: TypeAlias = int | float | bool | str
MetadataValue: TypeAlias = (
    Scalar | tuple[Scalar, ...] | tuple["MetadataValue", ...] | ArraySummary
)


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    n_elements: int
    n_bytes: int


@dataclass(frozen=True, slots=True)
class GGUFFile:
    path: Path
    version: int
    metadata: MappingProxyType[str, MetadataValue]
    tensors: tuple[TensorInfo, ...]
    alignment: int
    data_offset: int
    file_size: int


class _Reader:
    """Bounded little-endian reader that tracks its own position (no tell() syscalls)."""

    __slots__ = ("_fh", "_pos", "_size")

    def __init__(self, fh: BinaryIO, size: int) -> None:
        self._fh = fh
        self._size = size
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return self._size - self._pos

    def read_exact(self, n: int) -> bytes:
        if n < 0 or n > self.remaining:
            raise ModelFormatError(f"truncated GGUF: need {n} bytes at offset {self._pos}")
        data = self._fh.read(n)
        if len(data) != n:
            raise ModelFormatError(f"short read at offset {self._pos}")
        self._pos += n
        return data

    def skip(self, n: int) -> None:
        if n < 0 or n > self.remaining:
            raise ModelFormatError(f"truncated GGUF: cannot skip {n} bytes at {self._pos}")
        self._fh.seek(n, 1)
        self._pos += n

    def u32(self) -> int:
        value: int = _U32.unpack(self.read_exact(4))[0]
        return value

    def u64(self) -> int:
        value: int = _U64.unpack(self.read_exact(8))[0]
        return value

    def ensure_room(self, count: int, min_each: int, what: str) -> None:
        """Reject counts that the remaining file cannot possibly hold (anti-DoS)."""
        if count * min_each > self.remaining:
            raise ModelFormatError(f"{what} count {count} exceeds remaining file size")

    def string(self, max_bytes: int) -> str:
        length = self.u64()
        if length > max_bytes:
            raise ModelFormatError(f"string length {length} exceeds limit {max_bytes}")
        raw = self.read_exact(length)
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ModelFormatError(f"invalid UTF-8 string at offset {self._pos}") from exc

    def skip_string(self, max_bytes: int) -> None:
        length = self.u64()
        if length > max_bytes:
            raise ModelFormatError(f"string length {length} exceeds limit {max_bytes}")
        self.skip(length)

    def value_type(self) -> GGUFValueType:
        raw = self.u32()
        try:
            return GGUFValueType(raw)
        except ValueError as exc:
            raise ModelFormatError(f"unknown GGUF value type {raw}") from exc


def _read_scalar(r: _Reader, vtype: GGUFValueType) -> Scalar:
    fmt = _SCALAR_STRUCTS[vtype]
    raw = fmt.unpack(r.read_exact(fmt.size))[0]
    if vtype is GGUFValueType.BOOL:
        if raw not in (0, 1):
            raise ModelFormatError(f"invalid bool byte {raw}")
        return bool(raw)
    if vtype in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        return float(raw)
    return int(raw)


def _skip_array_elements(
    r: _Reader, etype: GGUFValueType, count: int, limits: ParserLimits, depth: int
) -> None:
    if etype is GGUFValueType.STRING:
        for _ in range(count):  # O(count) time, O(1) space
            r.skip_string(limits.max_string_bytes)
    elif etype is GGUFValueType.ARRAY:
        for _ in range(count):
            _read_array(r, limits, depth + 1, materialise=False)
    else:
        r.skip(count * _SCALAR_STRUCTS[etype].size)  # O(1)


def _read_array(
    r: _Reader, limits: ParserLimits, depth: int, *, materialise: bool = True
) -> MetadataValue:
    if depth >= limits.max_array_depth:
        raise ModelFormatError("GGUF array nesting exceeds limit")
    etype = r.value_type()
    count = r.u64()
    r.ensure_room(count, _MIN_ELEMENT_BYTES[etype], "array element")
    if not materialise or count > limits.max_inline_array:
        _skip_array_elements(r, etype, count, limits, depth)
        return ArraySummary(etype, count)
    if etype is GGUFValueType.STRING:
        return tuple(r.string(limits.max_string_bytes) for _ in range(count))
    if etype is GGUFValueType.ARRAY:
        return tuple(_read_array(r, limits, depth + 1) for _ in range(count))
    return tuple(_read_scalar(r, etype) for _ in range(count))


def _read_value(r: _Reader, vtype: GGUFValueType, limits: ParserLimits) -> MetadataValue:
    if vtype is GGUFValueType.STRING:
        return r.string(limits.max_string_bytes)
    if vtype is GGUFValueType.ARRAY:
        return _read_array(r, limits, depth=0)
    return _read_scalar(r, vtype)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _resolve_alignment(metadata: dict[str, MetadataValue]) -> int:
    raw = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ModelFormatError("general.alignment must be an integer")
    if raw <= 0 or raw > 1 << 16 or raw & (raw - 1):
        raise ModelFormatError(f"general.alignment must be a power of two <= 65536, got {raw}")
    return raw


def _expected_bytes(ggml_type: int, n_elements: int, name: str) -> int | None:
    layout = GGML_TYPE_LAYOUT.get(ggml_type)
    if layout is None:
        return None
    block, block_bytes = layout
    if n_elements % block:
        raise ModelFormatError(f"tensor {name!r}: {n_elements} elements not divisible by {block}")
    return n_elements // block * block_bytes


_RawTensor = tuple[str, tuple[int, ...], int, int, int]  # name, shape, type, offset, n_elem


def _read_metadata(r: _Reader, n_kv: int, limits: ParserLimits) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {}
    for _ in range(n_kv):
        key = r.string(limits.max_key_bytes)
        if not key:
            raise ModelFormatError("empty metadata key")
        if key in metadata:
            raise ModelFormatError(f"duplicate metadata key {key!r}")
        metadata[key] = _read_value(r, r.value_type(), limits)
    return metadata


def _read_tensor_info(r: _Reader, alignment: int, limits: ParserLimits) -> _RawTensor:
    name = r.string(limits.max_key_bytes)
    n_dims = r.u32()
    if not 1 <= n_dims <= GGML_MAX_DIMS:
        raise ModelFormatError(f"tensor {name!r}: invalid n_dims {n_dims}")
    shape = tuple(r.u64() for _ in range(n_dims))
    n_elements = 1
    for dim in shape:
        if not 1 <= dim <= limits.max_dim:
            raise ModelFormatError(f"tensor {name!r}: invalid dimension {dim}")
        n_elements *= dim
    if n_elements > limits.max_tensor_elements:
        raise ModelFormatError(f"tensor {name!r}: too many elements")
    ggml_type = r.u32()
    offset = r.u64()
    if offset % alignment:
        raise ModelFormatError(f"tensor {name!r}: misaligned offset {offset}")
    return name, shape, ggml_type, offset, n_elements


def _read_tensor_index(
    r: _Reader, n_tensors: int, alignment: int, limits: ParserLimits
) -> list[_RawTensor]:
    tensors: list[_RawTensor] = []
    seen: set[str] = set()
    for _ in range(n_tensors):
        info = _read_tensor_info(r, alignment, limits)
        if info[0] in seen:
            raise ModelFormatError(f"duplicate tensor name {info[0]!r}")
        seen.add(info[0])
        tensors.append(info)
    return tensors


def read_gguf(path: Path, limits: ParserLimits = DEFAULT_LIMITS) -> GGUFFile:
    """Parse the metadata and tensor index of a GGUF file without reading tensor data."""
    file_size = path.stat().st_size
    with path.open("rb", buffering=1 << 20) as fh:
        r = _Reader(fh, file_size)
        magic = r.read_exact(4)
        if magic == GGUF_MAGIC_BIG_ENDIAN:
            raise ModelFormatError("big-endian GGUF files are not supported")
        if magic != GGUF_MAGIC:
            raise ModelFormatError("not a GGUF file (bad magic)")
        version = r.u32()
        if version not in SUPPORTED_VERSIONS:
            raise ModelFormatError(f"unsupported GGUF version {version}")
        n_tensors = r.u64()
        n_kv = r.u64()
        if n_kv > limits.max_kv_count:
            raise ModelFormatError(f"metadata count {n_kv} exceeds limit")
        if n_tensors > limits.max_tensor_count:
            raise ModelFormatError(f"tensor count {n_tensors} exceeds limit")
        r.ensure_room(n_kv, 8 + 4 + 1, "metadata")
        r.ensure_room(n_tensors, 8 + 4 + 8 + 4 + 8, "tensor")

        metadata = _read_metadata(r, n_kv, limits)
        alignment = _resolve_alignment(metadata)
        raw_tensors = _read_tensor_index(r, n_tensors, alignment, limits)
        data_offset = _align_up(r.pos, alignment)
        if data_offset > file_size:
            raise ModelFormatError("tensor data section starts beyond end of file")
        data_size = file_size - data_offset

    tensors = _compute_extents(raw_tensors, data_size)
    return GGUFFile(
        path=path,
        version=version,
        metadata=MappingProxyType(metadata),
        tensors=tensors,
        alignment=alignment,
        data_offset=data_offset,
        file_size=file_size,
    )


def _compute_extents(raw: list[_RawTensor], data_size: int) -> tuple[TensorInfo, ...]:
    """Derive each tensor's byte size from offsets; reject overlaps / out-of-range data.

    O(T log T) for the sort; O(T) space.
    """
    order = sorted(range(len(raw)), key=lambda i: raw[i][3])
    sizes = [0] * len(raw)
    for rank, idx in enumerate(order):
        name, _shape, ggml_type, offset, n_elements = raw[idx]
        end = raw[order[rank + 1]][3] if rank + 1 < len(order) else data_size
        if offset >= data_size or end <= offset:
            raise ModelFormatError(f"tensor {name!r}: invalid or overlapping data offset")
        span = end - offset
        expected = _expected_bytes(ggml_type, n_elements, name)
        if expected is not None and expected > span:
            raise ModelFormatError(f"tensor {name!r}: data ({expected} B) exceeds its extent")
        sizes[idx] = expected if expected is not None else span
    return tuple(
        TensorInfo(name, shape, ggml_type, offset, n_elements, sizes[i])
        for i, (name, shape, ggml_type, offset, n_elements) in enumerate(raw)
    )


# ---- typed metadata accessors -------------------------------------------------------------


def meta_int(metadata: MappingProxyType[str, MetadataValue], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelFormatError(f"metadata {key!r} must be an integer")
    return value


def meta_int_or_max(metadata: MappingProxyType[str, MetadataValue], key: str) -> int | None:
    """Some architectures store per-layer values as arrays; return the scalar or the max."""
    value = metadata.get(key)
    if isinstance(value, tuple) and value:
        ints = [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
        if len(ints) != len(value):
            raise ModelFormatError(f"metadata {key!r} must be an integer array")
        return max(ints)
    return meta_int(metadata, key)


def meta_str(metadata: MappingProxyType[str, MetadataValue], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelFormatError(f"metadata {key!r} must be a string")
    return value


def meta_array_len(metadata: MappingProxyType[str, MetadataValue], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, ArraySummary):
        return value.length
    if isinstance(value, tuple):
        return len(value)
    return None
