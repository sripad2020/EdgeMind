from __future__ import annotations

import struct
from pathlib import Path

import pytest

from air_sdk.analysis.gguf import ArraySummary, ParserLimits, read_gguf
from air_sdk.analysis.profile import analyze_model, discover_variants
from air_sdk.analysis.quant import KVCacheType
from air_sdk.errors import ModelFormatError
from air_sdk.testing import SyntheticModel, write_gguf


def test_parse_profile(model_path: Path) -> None:
    p = analyze_model(model_path)
    m = SyntheticModel()
    assert p.architecture == "llama"
    assert p.n_layers == m.n_layers
    assert p.n_embd == m.n_embd
    assert p.n_head_kv == m.n_head_kv
    assert p.n_vocab == m.n_vocab
    assert p.quant.name == "F16"
    # Per-layer bytes + suffix sums are consistent.
    assert p.layer_suffix_bytes[0] == 0
    assert p.layer_suffix_bytes[-1] == sum(p.layer_bytes)
    assert p.weight_bytes == sum(p.layer_bytes) + p.output_bytes + p.other_bytes
    # Weight bytes cannot exceed the file.
    assert p.weight_bytes <= p.file_size
    head_dim = m.n_embd // m.n_head
    assert p.kv_bytes_per_token(KVCacheType.F16) == m.n_layers * m.n_head_kv * 2 * head_dim * 2


def test_large_array_is_summarised_not_materialised(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "v.gguf", SyntheticModel(n_vocab=512))
    g = read_gguf(path, ParserLimits(max_inline_array=16))
    tokens = g.metadata["tokenizer.ggml.tokens"]
    assert isinstance(tokens, ArraySummary)
    assert tokens.length == 512


def test_discover_variants(model_dir: Path) -> None:
    primary = analyze_model(model_dir / "tiny-f16.gguf")
    names = sorted(v.quant.name for v in discover_variants(primary))
    assert names == ["F16", "Q4_0", "Q8_0"]


def test_bad_magic(tmp_path: Path) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 60)
    with pytest.raises(ModelFormatError, match="magic"):
        read_gguf(p)


def test_truncated_file(model_path: Path, tmp_path: Path) -> None:
    data = model_path.read_bytes()
    t = tmp_path / "trunc.gguf"
    t.write_bytes(data[:200])
    with pytest.raises(ModelFormatError):
        read_gguf(t)


def test_hostile_counts_rejected_before_allocation(tmp_path: Path) -> None:
    p = tmp_path / "hostile.gguf"
    # Claims 2^40 metadata entries in a 64-byte file.
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1 << 40) + b"\0" * 40)
    with pytest.raises(ModelFormatError):
        read_gguf(p)
    # Claims a 1 TiB string.
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1 << 40)
                  + b"\0" * 40)
    with pytest.raises(ModelFormatError, match="exceeds limit"):
        read_gguf(p)


def test_unsupported_version(tmp_path: Path) -> None:
    p = tmp_path / "v1.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 1, 0, 0) + b"\0" * 40)
    with pytest.raises(ModelFormatError, match="version"):
        read_gguf(p)


def test_tensor_beyond_file_rejected(model_path: Path, tmp_path: Path) -> None:
    data = model_path.read_bytes()
    t = tmp_path / "short-data.gguf"
    t.write_bytes(data[: len(data) - 4096])  # chop tensor data
    with pytest.raises(ModelFormatError):
        analyze_model(t)
