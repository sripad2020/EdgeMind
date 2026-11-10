"""Engine 3: Dynamic CPU-GPU placement (deterministic estimation + candidate generation).

llama.cpp offloads the *last* ``n_gpu_layers`` repeating layers to the GPU, and the output
layer as well when ``n_gpu_layers > n_layers``. Placement levels therefore range over
``k in [0, n_layers + 1]``.

Complexity
  estimate_memory: O(1) thanks to the precomputed per-layer suffix sums in ModelProfile.
  max_fitting_layers: O(log L) binary search (VRAM use is monotone non-decreasing in k).
"""

from __future__ import annotations

from typing import Final

from .._validate import MiB
from ..analysis.profile import ModelProfile
from ..analysis.quant import KVCacheType
from ..config import MemoryEstimate

CUDA_CONTEXT_BYTES: Final[int] = 384 * MiB
RUNTIME_RAM_OVERHEAD: Final[int] = 256 * MiB
_F32: Final[int] = 4
_FLASH_ATTN_TILE: Final[int] = 256
_LOGIT_ROWS: Final[int] = 8


def compute_buffer_bytes(p: ModelProfile, n_ctx: int, n_batch: int, flash_attn: bool) -> int:
    """Conservative estimate of the scratch/compute buffer for one ubatch."""
    activations = _F32 * n_batch * (4 * p.n_embd + max(p.n_ff, p.n_embd))
    attn_cols = _FLASH_ATTN_TILE if flash_attn else n_ctx
    attention = _F32 * n_batch * attn_cols * p.n_head
    # Generation only materialises logits for the last token(s) of a ubatch.
    logits = _F32 * p.n_vocab * min(n_batch, _LOGIT_ROWS)
    return activations + attention + logits


def estimate_memory(
    p: ModelProfile,
    *,
    n_gpu_layers: int,
    kv_type: KVCacheType,
    n_ctx: int,
    n_batch: int,
    flash_attn: bool,
    gpu_context_bytes: int = CUDA_CONTEXT_BYTES,
    runtime_bytes: int = RUNTIME_RAM_OVERHEAD,
    overhead_scale: float = 1.0,
) -> MemoryEstimate:
    """Weights are counted exactly (from the tensor index); everything else (KV cache,
    compute buffers, runtime reserves) is multiplied by ``overhead_scale``, a per-machine
    calibration factor derived from measured footprints (1.0 = uncalibrated, conservative).
    """
    gpu_layers = min(n_gpu_layers, p.n_layers)
    offload_output = n_gpu_layers > p.n_layers
    gpu_weights = p.layer_suffix_bytes[gpu_layers] + (p.output_bytes if offload_output else 0)
    cpu_weights = p.weight_bytes - gpu_weights
    kv_gpu = int(p.kv_bytes_per_token(kv_type, gpu_layers) * n_ctx)
    kv_cpu = int(p.kv_bytes_per_token(kv_type, p.n_layers - gpu_layers) * n_ctx)
    compute = compute_buffer_bytes(p, n_ctx, n_batch, flash_attn)
    vram_overhead = kv_gpu + (compute + gpu_context_bytes if n_gpu_layers > 0 else 0)
    ram_overhead = kv_cpu + compute + runtime_bytes
    return MemoryEstimate(
        ram_bytes=cpu_weights + int(ram_overhead * overhead_scale),
        vram_bytes=gpu_weights + int(vram_overhead * overhead_scale),
    )


def max_fitting_layers(
    p: ModelProfile,
    *,
    vram_budget: int,
    kv_type: KVCacheType,
    n_ctx: int,
    n_batch: int,
    flash_attn: bool,
    gpu_context_bytes: int = CUDA_CONTEXT_BYTES,
    runtime_bytes: int = RUNTIME_RAM_OVERHEAD,
) -> int:
    """Largest k in [0, L+1] whose VRAM estimate fits ``vram_budget`` (0 if none)."""
    lo, hi = 0, p.n_layers + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        est = estimate_memory(
            p, n_gpu_layers=mid, kv_type=kv_type, n_ctx=n_ctx, n_batch=n_batch,
            flash_attn=flash_attn, gpu_context_bytes=gpu_context_bytes,
            runtime_bytes=runtime_bytes,
        )
        if est.vram_bytes <= vram_budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def placement_levels(p: ModelProfile, k_max: int, n_levels: int = 5) -> tuple[int, ...]:
    """Evenly spaced offload levels in [0, k_max] (always includes 0 and k_max)."""
    if k_max <= 0:
        return (0,)
    full = p.n_layers + 1
    levels = {0, k_max}
    for i in range(1, n_levels - 1):
        k = round(full * i / (n_levels - 1))
        if 0 < k < k_max:
            levels.add(k)
    return tuple(sorted(levels))
