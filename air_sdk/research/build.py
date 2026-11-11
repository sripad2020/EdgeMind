"""Build the execution profiles under study as Ollama models (base weights never edited).

* uniform      - whole model at Q8_0 or Q4_0 (same names as ``make_faster``)
* sensitivity  - one layer group at Q4_0, everything else at Q8_0          (RQ1)
* mix          - chosen layer groups kept at Q8_0, everything else Q4_0     (RQ2)
* adapter      - a LoRA adapter (GGUF) attached to the base or any variant  (RQ3)

Every file is written by EdgeMind's own quantizer from the F16 source, validated with the
hardened GGUF parser, uploaded to Ollama's blob store (SHA-256 verified), registered with
the source's chat template and parameters, then deleted locally (Ollama keeps a copy).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

from ..analysis.gguf import TensorInfo, read_gguf
from ..analysis.profile import ModelProfile, analyze_model
from ..backends.ollama import OllamaClient, OllamaModelInfo, validate_model_name
from ..errors import ValidationError
from ..quantize import QuantTarget, derived_model_name, is_quantizable, quantize_gguf
from .store import Profile

_BLOCK: Final = re.compile(r"^blk\.(\d+)\.")
_LABEL: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,23}")


def layer_of(tensor: TensorInfo) -> int | None:
    m = _BLOCK.match(tensor.name)
    return int(m.group(1)) if m else None


def n_groups(n_layers: int, group_size: int) -> int:
    if group_size < 1:
        raise ValidationError("group size must be at least 1")
    return (n_layers + group_size - 1) // group_size


def group_override(groups: set[int], group_size: int,
                   inside: QuantTarget) -> Callable[[TensorInfo, QuantTarget], QuantTarget]:
    """Tensors of layers in ``groups`` get ``inside``; every other tensor keeps its default."""
    def choose(t: TensorInfo, default: QuantTarget) -> QuantTarget:
        layer = layer_of(t)
        return inside if layer is not None and layer // group_size in groups else default
    return choose


def _derived(base: str, label: str) -> str:
    return validate_model_name(f"{base}-air-{label}" if ":" in base else f"{base}:air-{label}")


def _digest_of(info: OllamaModelInfo) -> str:
    m = re.fullmatch(r"sha256[-:]([0-9a-f]{64})", info.blob_path.name)
    if not m:
        raise ValidationError(f"cannot find the blob digest of {info.name}")
    return f"sha256:{m.group(1)}"


class Builder:
    def __init__(self, client: OllamaClient, workdir: Path) -> None:
        self.client = client
        self.workdir = workdir
        workdir.mkdir(parents=True, exist_ok=True)

    # -- helpers -----------------------------------------------------------------------------

    def source(self, base: str) -> tuple[OllamaModelInfo, ModelProfile]:
        info = self.client.show(validate_model_name(base))
        profile = analyze_model(info.blob_path, suffixes=("", ".gguf"))
        return info, profile

    def _installed(self, name: str) -> bool:
        return name in set(self.client.list_models())

    def _size(self, name: str) -> int:
        return analyze_model(self.client.show(name).blob_path, suffixes=("", ".gguf")).file_size

    def _quantized(self, base: str, name: str, target: QuantTarget,
                   override: Callable[[TensorInfo, QuantTarget], QuantTarget] | None) -> int:
        if self._installed(name):
            return self._size(name)
        info, profile = self.source(base)
        if not is_quantizable(profile):
            raise ValidationError(f"{base} is {profile.quant.name}; the study needs an F16/BF16 "
                                  "source (e.g. pull an -fp16 tag)")
        dst = self.workdir / f"{hashlib.sha256(name.encode()).hexdigest()[:16]}.gguf"
        try:
            made = quantize_gguf(profile.path, dst, target, override=override)
            digest = self.client.push_blob(dst)
            self.client.create_model(name, gguf_digest=digest, filename="model.gguf",
                                     settings=self.client.model_settings(info.name))
            return made.file_size
        finally:
            if dst.exists():
                dst.unlink()

    # -- the profile kinds -------------------------------------------------------------------

    def baseline(self, base: str) -> Profile:
        _info, profile = self.source(base)
        return Profile(base, base, "model", profile.quant.name, size_bytes=profile.file_size,
                       baseline=True)

    def uniform(self, base: str, target: QuantTarget) -> Profile:
        name = derived_model_name(base, target)
        size = self._quantized(base, name, target, None)
        return Profile(name, base, "uniform", target.name, size_bytes=size)

    def sensitivity(self, base: str, group_size: int) -> list[Profile]:
        _info, profile = self.source(base)
        out = []
        for g in range(n_groups(profile.n_layers, group_size)):
            name = _derived(base, f"sens{group_size}-g{g}")
            size = self._quantized(base, name, QuantTarget.Q8_0,
                                   group_override({g}, group_size, QuantTarget.Q4_0))
            out.append(Profile(name, base, "sensitivity", "Q8_0 with one group Q4_0",
                               groups=str(g), size_bytes=size,
                               meta=json.dumps({"group_size": group_size, "group": g})))
        return out

    def mix(self, base: str, keep: set[int], group_size: int, label: str) -> Profile:
        if not _LABEL.fullmatch(label):
            raise ValidationError("label must be 1-24 lower-case letters, digits or '-'")
        name = _derived(base, f"mix-{label}")
        size = self._quantized(base, name, QuantTarget.Q4_0,
                               group_override(keep, group_size, QuantTarget.Q8_0))
        groups = ",".join(str(g) for g in sorted(keep))
        return Profile(name, base, "mix", "Q4_0 with groups Q8_0", groups=groups,
                       size_bytes=size,
                       meta=json.dumps({"group_size": group_size, "keep_q8": sorted(keep)}))

    def adapter(self, variant: str, adapter_path: Path, label: str, base: str) -> Profile:
        """Attach a GGUF LoRA adapter to ``variant`` (the base model or one of its builds)."""
        if not _LABEL.fullmatch(label):
            raise ValidationError("label must be 1-24 lower-case letters, digits or '-'")
        target_info, target_profile = self.source(variant)
        gguf = read_gguf(adapter_path)  # hardened parse before anything is uploaded
        meta = gguf.metadata
        if meta.get("general.type") != "adapter" or meta.get("adapter.type") != "lora":
            raise ValidationError(f"{adapter_path.name} is not a GGUF LoRA adapter")
        arch = meta.get("general.architecture")
        if arch != target_profile.architecture:
            raise ValidationError(f"adapter is for {arch!r}, base model is "
                                  f"{target_profile.architecture!r}")
        name = validate_model_name(f"{variant}-lora-{label}" if ":" in variant
                                   else f"{variant}:lora-{label}")
        if not self._installed(name):
            digest = self.client.push_blob(adapter_path)
            self.client.create_model(
                name, gguf_digest=_digest_of(target_info), filename="model.gguf",
                settings=self.client.model_settings(target_info.name),
                adapters={"adapter.gguf": digest})
        return Profile(name, base, "adapter", target_profile.quant.name, adapter=label,
                       size_bytes=target_profile.file_size + adapter_path.stat().st_size,
                       meta=json.dumps({"on": variant}))

    def remove(self, name: str) -> None:
        self.client.delete_model(name)
