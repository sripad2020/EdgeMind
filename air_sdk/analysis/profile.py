"""Engine 1: Intelligent Model Analysis -> Model Intelligence Profile.

Purely deterministic: parses GGUF metadata and the tensor index, then computes parameter
counts, per-layer weight sizes and KV-cache cost per token.

Complexity: dominated by :func:`read_gguf` (see there). Profile construction is O(T) over
tensors; per-layer prefix sums are precomputed once so later placement queries for
"bytes in the last k layers" are O(1).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from .._validate import GiB, resolve_file
from ..errors import ModelFormatError
from .gguf import (
    DEFAULT_LIMITS,
    GGUFFile,
    ParserLimits,
    meta_array_len,
    meta_int,
    meta_int_or_max,
    meta_str,
    read_gguf,
)
from .quant import (
    KVCacheType,
    WeightQuant,
    quant_from_dominant_ggml_type,
    quant_from_file_type,
)

MAX_MODEL_FILE_BYTES: Final[int] = 1024 * GiB
_ARCH_RE: Final = re.compile(r"^[a-z0-9_.\-]{1,64}$")
_BLOCK_RE: Final = re.compile(r"^blk\.(\d{1,6})\.")


class ModelSizeClass(str, Enum):
    SMALL = "small"  # < 3B parameters
    MEDIUM = "medium"  # < 14B
    LARGE = "large"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Model Intelligence Profile."""

    path: Path
    file_size: int
    architecture: str
    name: str
    quant: WeightQuant
    n_params: int
    n_layers: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim_k: int
    head_dim_v: int
    n_ff: int
    n_vocab: int
    context_length: int
    expert_count: int
    weight_bytes: int
    layer_bytes: tuple[int, ...]
    # layer_suffix_bytes[k] = bytes of the last k repeating layers (k in 0..n_layers)
    layer_suffix_bytes: tuple[int, ...]
    output_bytes: int
    other_bytes: int
    family_key: str
    variant_key: str

    @property
    def attention_free(self) -> bool:
        """True for architectures without attention (e.g. Mamba/RWKV): no growing KV cache."""
        return self.n_head == 0

    @property
    def size_class(self) -> ModelSizeClass:
        if self.n_params < 3_000_000_000:
            return ModelSizeClass.SMALL
        if self.n_params < 14_000_000_000:
            return ModelSizeClass.MEDIUM
        return ModelSizeClass.LARGE

    def kv_bytes_per_token(self, kv_type: KVCacheType, n_layers: int | None = None) -> float:
        """KV-cache bytes for one token across ``n_layers`` layers (default: all)."""
        layers = self.n_layers if n_layers is None else n_layers
        per_layer = self.n_head_kv * (self.head_dim_k + self.head_dim_v)
        return layers * per_layer * kv_type.bytes_per_element

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "architecture": self.architecture,
            "name": self.name,
            "precision": self.quant.name,
            "parameters": self.n_params,
            "layers": self.n_layers,
            "hidden_size": self.n_embd,
            "attention_heads": self.n_head,
            "kv_heads": self.n_head_kv,
            "feed_forward": self.n_ff,
            "vocab": self.n_vocab,
            "max_context": self.context_length,
            "experts": self.expert_count,
            "weight_bytes": self.weight_bytes,
            "kv_bytes_per_token_f16": round(self.kv_bytes_per_token(KVCacheType.F16)),
            "size_class": self.size_class.value,
        }


def _require(value: int | None, key: str, *, minimum: int = 1) -> int:
    if value is None:
        raise ModelFormatError(f"required metadata {key!r} is missing")
    if value < minimum:
        raise ModelFormatError(f"metadata {key!r} must be >= {minimum}, got {value}")
    return value


def _hash(*parts: object) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


_MAX_LAYERS: Final[int] = 100_000


def _infer_layers(gguf: GGUFFile) -> int | None:
    """Number of repeating blocks = highest ``blk.N.`` index + 1 (None if there are none)."""
    highest = -1
    for t in gguf.tensors:
        m = _BLOCK_RE.match(t.name)
        if m is not None:
            highest = max(highest, int(m.group(1)))
    return highest + 1 if 0 <= highest < _MAX_LAYERS else None


def _infer_embd(gguf: GGUFFile) -> int | None:
    """Hidden size from the token-embedding matrix ([n_embd, n_vocab] in GGUF order)."""
    for t in gguf.tensors:
        if t.name == "token_embd.weight" and len(t.shape) == 2:
            return t.shape[0]
    return None


def build_profile(gguf: GGUFFile) -> ModelProfile:
    md = gguf.metadata
    arch = meta_str(md, "general.architecture")
    if arch is None or not _ARCH_RE.fullmatch(arch):
        raise ModelFormatError("general.architecture missing or invalid")
    name = (meta_str(md, "general.name") or gguf.path.stem)[:256]

    # Fields are read from metadata when present and otherwise derived from the tensors, so
    # uncommon architectures (attention-free Mamba/RWKV, hybrids, partial metadata) still work.
    n_layers = _require(meta_int(md, f"{arch}.block_count") or _infer_layers(gguf),
                        f"{arch}.block_count")
    n_embd = _require(meta_int(md, f"{arch}.embedding_length") or _infer_embd(gguf),
                      f"{arch}.embedding_length")
    # 0 attention heads = attention-free model: no KV cache grows with the context.
    n_head = meta_int_or_max(md, f"{arch}.attention.head_count") or 0
    n_head_kv = meta_int_or_max(md, f"{arch}.attention.head_count_kv") or n_head
    default_head_dim = n_embd // n_head if n_head else 0
    head_dim_k = meta_int(md, f"{arch}.attention.key_length") or default_head_dim
    head_dim_v = meta_int(md, f"{arch}.attention.value_length") or default_head_dim
    n_ff = meta_int_or_max(md, f"{arch}.feed_forward_length") or 0
    context_length = meta_int(md, f"{arch}.context_length") or 2048
    expert_count = meta_int(md, f"{arch}.expert_count") or 0

    layer_bytes = [0] * n_layers
    output_bytes = 0
    other_bytes = 0
    n_params = 0
    bytes_by_type: dict[int, int] = {}
    vocab_from_tensor: int | None = None
    for t in gguf.tensors:
        n_params += t.n_elements
        bytes_by_type[t.ggml_type] = bytes_by_type.get(t.ggml_type, 0) + t.n_bytes
        match = _BLOCK_RE.match(t.name)
        if match is not None:
            idx = int(match.group(1))
            if idx >= n_layers:
                raise ModelFormatError(f"tensor {t.name!r} references layer >= block_count")
            layer_bytes[idx] += t.n_bytes
        elif t.name.startswith("output"):
            output_bytes += t.n_bytes
        else:
            other_bytes += t.n_bytes
            if t.name == "token_embd.weight" and len(t.shape) == 2:
                vocab_from_tensor = t.shape[1]

    n_vocab = (
        meta_array_len(md, "tokenizer.ggml.tokens")
        or meta_int(md, f"{arch}.vocab_size")
        or vocab_from_tensor
        or 0
    )

    suffix = [0] * (n_layers + 1)
    for k in range(1, n_layers + 1):  # O(L) once; O(1) per later query
        suffix[k] = suffix[k - 1] + layer_bytes[n_layers - k]

    file_type = meta_int(md, "general.file_type")
    quant = quant_from_file_type(file_type)
    if file_type is None and bytes_by_type:
        dominant = max(bytes_by_type.items(), key=lambda kv: kv[1])[0]
        quant = quant_from_dominant_ggml_type(dominant)

    weight_bytes = sum(layer_bytes) + output_bytes + other_bytes
    family_key = _hash(arch, n_layers, n_embd, n_head, n_head_kv, n_ff, n_vocab, n_params)
    variant_key = _hash(family_key, quant.name, gguf.file_size, name)

    return ModelProfile(
        path=gguf.path,
        file_size=gguf.file_size,
        architecture=arch,
        name=name,
        quant=quant,
        n_params=n_params,
        n_layers=n_layers,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        n_ff=n_ff,
        n_vocab=n_vocab,
        context_length=context_length,
        expert_count=expert_count,
        weight_bytes=weight_bytes,
        layer_bytes=tuple(layer_bytes),
        layer_suffix_bytes=tuple(suffix),
        output_bytes=output_bytes,
        other_bytes=other_bytes,
        family_key=family_key,
        variant_key=variant_key,
    )


def analyze_model(
    path: object,
    limits: ParserLimits = DEFAULT_LIMITS,
    *,
    suffixes: tuple[str, ...] = (".gguf",),
) -> ModelProfile:
    """Validate ``path`` and build its Model Intelligence Profile.

    ``suffixes`` may include ``""`` for content-addressed blobs (e.g. Ollama's
    ``sha256-...`` files); the GGUF magic/version checks still apply.
    """
    resolved = resolve_file(
        "model path", path, suffixes=suffixes, min_bytes=24, max_bytes=MAX_MODEL_FILE_BYTES
    )
    return build_profile(read_gguf(resolved, limits))


def discover_variants(
    primary: ModelProfile, *, max_files: int = 64, limits: ParserLimits = DEFAULT_LIMITS
) -> tuple[ModelProfile, ...]:
    """Find other GGUF files in the same directory that are the same model at another
    precision (same ``family_key``). Unparseable files are skipped, never fatal.

    O(F) header parses, F <= ``max_files``.
    """
    found: dict[str, ModelProfile] = {primary.variant_key: primary}
    candidates = sorted(p for p in primary.path.parent.glob("*.gguf") if p != primary.path)
    for candidate in candidates[:max_files]:
        try:
            profile = analyze_model(candidate, limits)
        except (ModelFormatError, OSError, ValueError):
            continue
        if profile.family_key == primary.family_key:
            found.setdefault(profile.variant_key, profile)
    return tuple(found.values())
