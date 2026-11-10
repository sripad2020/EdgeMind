"""Quantizer tests: kernels vs a literal port of llama.cpp's reference code, GGUF round-trip."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from air_sdk.analysis.gguf import read_gguf
from air_sdk.analysis.profile import analyze_model
from air_sdk.backends.ollama import _parse_parameters
from air_sdk.quantize import (
    QuantizationError,
    QuantTarget,
    dequantize_q4_0,
    dequantize_q8_0,
    derived_model_name,
    plan_targets,
    quantize_gguf,
    quantize_q4_0,
    quantize_q8_0,
)
from air_sdk.testing import SyntheticModel, write_gguf


def _f16(x: float) -> float:
    return float(np.float16(x))


def _ref_q8_0(block: list[float]) -> bytes:
    """Literal port of llama.cpp quantize_row_q8_0_ref for one block."""
    amax = max(abs(v) for v in block)
    d = amax / 127.0
    inv = 1.0 / d if d else 0.0
    qs = [int(math.copysign(math.floor(abs(v * inv) + 0.5), v * inv)) for v in block]
    return struct.pack("<e", d) + struct.pack("<32b", *qs)


def _ref_q4_0(block: list[float]) -> bytes:
    """Literal port of llama.cpp quantize_row_q4_0_ref for one block."""
    amax, mx = 0.0, 0.0
    for v in block:
        if amax < abs(v):
            amax, mx = abs(v), v
    d = mx / -8.0
    inv = 1.0 / d if d else 0.0
    qs = []
    for j in range(16):
        x0 = min(15, int(block[j] * inv + 8.5))
        x1 = min(15, int(block[16 + j] * inv + 8.5))
        qs.append(x0 | (x1 << 4))
    return struct.pack("<e", d) + bytes(qs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernels_match_llama_cpp_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(32 * 8) * rng.uniform(0.01, 3)).astype(np.float32)
    data[5] = 0.0
    q8 = quantize_q8_0(data.reshape(8, 32))
    q4 = quantize_q4_0(data.reshape(8, 32))
    blocks = [data[i * 32:(i + 1) * 32].tolist() for i in range(8)]
    assert q8 == b"".join(_ref_q8_0(b) for b in blocks)
    assert q4 == b"".join(_ref_q4_0(b) for b in blocks)


def test_all_zero_block_is_safe() -> None:
    z = np.zeros((1, 32), dtype=np.float32)
    assert np.all(dequantize_q8_0(quantize_q8_0(z)) == 0)
    assert np.all(dequantize_q4_0(quantize_q4_0(z)) == 0)


def test_roundtrip_error_is_bounded() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((64, 256)).astype(np.float32)
    for kernel, dekernel, max_rel in ((quantize_q8_0, dequantize_q8_0, 0.01),
                                      (quantize_q4_0, dequantize_q4_0, 0.15)):
        y = dekernel(kernel(x)).reshape(x.shape)
        rel = float(np.linalg.norm(y - x) / np.linalg.norm(x))
        assert rel < max_rel, (kernel.__name__, rel)


@pytest.mark.parametrize("target", [QuantTarget.Q8_0, QuantTarget.Q4_0])
def test_quantize_gguf_roundtrip(tmp_path: Path, target: QuantTarget) -> None:
    src = write_gguf(tmp_path / "src.gguf", SyntheticModel(file_type=1))
    dst = tmp_path / f"out-{target.value}.gguf"
    profile = quantize_gguf(src, dst, target)
    assert profile.quant.name == target.value.upper()
    assert profile.family_key == analyze_model(src).family_key  # same model
    g = read_gguf(dst)
    types = {t.name: t.ggml_type for t in g.tensors}
    assert types["blk.0.attn_norm.weight"] == 0  # 1-D norms stay F32
    assert types["blk.0.ffn_up.weight"] == target.ggml_type
    if target is QuantTarget.Q4_0:  # quality-sensitive tensors kept at Q8_0
        assert types["token_embd.weight"] == 8
        assert types["output.weight"] == 8
    assert dst.stat().st_size < src.stat().st_size
    assert not list(tmp_path.glob("*.partial"))  # atomic write leaves no temp file


def test_refuses_already_quantized_source(tmp_path: Path) -> None:
    src = write_gguf(tmp_path / "q4.gguf", SyntheticModel(file_type=2))
    with pytest.raises(QuantizationError):
        quantize_gguf(src, tmp_path / "x.gguf", QuantTarget.Q8_0)
    assert plan_targets(analyze_model(src), 0.05) == []


def test_plan_respects_quality_limit(tmp_path: Path) -> None:
    src = analyze_model(write_gguf(tmp_path / "f16.gguf", SyntheticModel(file_type=1)))
    assert [p.target for p in plan_targets(src, 0.05)] == [QuantTarget.Q8_0, QuantTarget.Q4_0]
    assert [p.target for p in plan_targets(src, 0.01)] == [QuantTarget.Q8_0]


def test_derived_names() -> None:
    assert derived_model_name("smollm2:360m", QuantTarget.Q8_0) == "smollm2:360m-air-q8_0"
    assert derived_model_name("phi3", QuantTarget.Q4_0) == "phi3:air-q4_0"


def test_parse_ollama_parameters() -> None:
    text = 'stop                           "<|im_start|>"\nstop "<|im_end|>"\n' \
           "temperature 0.2\nnum_ctx 4096\nbad line here\n"
    assert _parse_parameters(text) == {
        "stop": ["<|im_start|>", "<|im_end|>"], "temperature": 0.2, "num_ctx": 4096,
    }
