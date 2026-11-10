"""Test utilities: write small, structurally valid GGUF files (zero-filled weights).

Used by the test-suite and the simulated demo. The produced files contain no real weights
and must never be used for inference with a real backend.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .analysis.gguf import GGML_TYPE_LAYOUT, GGUFValueType

_FTYPE_TO_GGML = {0: 0, 1: 1, 32: 30, 7: 8, 2: 2, 18: 14, 15: 12}  # file_type -> tensor type


@dataclass(frozen=True, slots=True)
class SyntheticModel:
    name: str = "synthetic-llama"
    arch: str = "llama"
    n_layers: int = 4
    n_embd: int = 128
    n_head: int = 4
    n_head_kv: int = 2
    n_ff: int = 384
    n_vocab: int = 512
    context_length: int = 4096
    file_type: int = 1  # F16


def _w_str(f: BinaryIO, s: str) -> None:
    raw = s.encode("utf-8")
    f.write(struct.pack("<Q", len(raw)))
    f.write(raw)


def _w_kv(f: BinaryIO, key: str, vtype: GGUFValueType, value: object) -> None:
    _w_str(f, key)
    f.write(struct.pack("<I", vtype))
    if vtype is GGUFValueType.STRING:
        assert isinstance(value, str)  # noqa: S101
        _w_str(f, value)
    elif vtype is GGUFValueType.UINT32:
        f.write(struct.pack("<I", value))
    elif vtype is GGUFValueType.ARRAY:
        assert isinstance(value, list)  # noqa: S101
        f.write(struct.pack("<I", GGUFValueType.STRING))
        f.write(struct.pack("<Q", len(value)))
        for item in value:
            _w_str(f, str(item))
    else:
        raise ValueError(f"unsupported writer type {vtype}")


def _nbytes(ggml_type: int, n_elements: int) -> int:
    block, size = GGML_TYPE_LAYOUT[ggml_type]
    return n_elements // block * size


def write_gguf(path: Path, model: SyntheticModel, *, alignment: int = 32,
               omit_keys: tuple[str, ...] = ()) -> Path:
    """Write a synthetic GGUF v3 file. Returns ``path``. ``omit_keys`` drops metadata keys
    (e.g. to emulate architectures/files without attention metadata)."""
    weight_type = _FTYPE_TO_GGML[model.file_type]
    head_dim = model.n_embd // model.n_head
    kv_dim = head_dim * model.n_head_kv
    tensors: list[tuple[str, tuple[int, ...], int]] = [
        ("token_embd.weight", (model.n_embd, model.n_vocab), weight_type),
    ]
    for i in range(model.n_layers):
        tensors += [
            (f"blk.{i}.attn_norm.weight", (model.n_embd,), 0),
            (f"blk.{i}.attn_q.weight", (model.n_embd, model.n_embd), weight_type),
            (f"blk.{i}.attn_k.weight", (model.n_embd, kv_dim), weight_type),
            (f"blk.{i}.attn_v.weight", (model.n_embd, kv_dim), weight_type),
            (f"blk.{i}.attn_output.weight", (model.n_embd, model.n_embd), weight_type),
            (f"blk.{i}.ffn_norm.weight", (model.n_embd,), 0),
            (f"blk.{i}.ffn_gate.weight", (model.n_embd, model.n_ff), weight_type),
            (f"blk.{i}.ffn_up.weight", (model.n_embd, model.n_ff), weight_type),
            (f"blk.{i}.ffn_down.weight", (model.n_ff, model.n_embd), weight_type),
        ]
    tensors += [
        ("output_norm.weight", (model.n_embd,), 0),
        ("output.weight", (model.n_embd, model.n_vocab), weight_type),
    ]
    a = model.arch
    kvs: list[tuple[str, GGUFValueType, object]] = [
        ("general.architecture", GGUFValueType.STRING, a),
        ("general.name", GGUFValueType.STRING, model.name),
        ("general.file_type", GGUFValueType.UINT32, model.file_type),
        ("general.alignment", GGUFValueType.UINT32, alignment),
        (f"{a}.block_count", GGUFValueType.UINT32, model.n_layers),
        (f"{a}.context_length", GGUFValueType.UINT32, model.context_length),
        (f"{a}.embedding_length", GGUFValueType.UINT32, model.n_embd),
        (f"{a}.feed_forward_length", GGUFValueType.UINT32, model.n_ff),
        (f"{a}.attention.head_count", GGUFValueType.UINT32, model.n_head),
        (f"{a}.attention.head_count_kv", GGUFValueType.UINT32, model.n_head_kv),
        ("tokenizer.ggml.tokens", GGUFValueType.ARRAY, [f"t{i}" for i in range(model.n_vocab)]),
    ]
    kvs = [kv for kv in kvs if kv[0] not in omit_keys]
    offsets: list[int] = []
    cursor = 0
    for _name, shape, ttype in tensors:
        n = 1
        for d in shape:
            n *= d
        offsets.append(cursor)
        cursor += _nbytes(ttype, n)
        cursor = (cursor + alignment - 1) // alignment * alignment

    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(tensors), len(kvs)))
        for key, vtype, value in kvs:
            _w_kv(f, key, vtype, value)
        for (name, shape, ttype), off in zip(tensors, offsets, strict=True):
            _w_str(f, name)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<IQ", ttype, off))
        pad = (-f.tell()) % alignment
        f.write(b"\0" * pad)
        remaining = cursor
        chunk = b"\0" * (1 << 20)
        while remaining > 0:
            n = min(remaining, len(chunk))
            f.write(chunk[:n])
            remaining -= n
    return path
