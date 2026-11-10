"""AIR-SDK demo.

    python main.py                         # simulated GPU machine + synthetic model
    python main.py --model path/to/x.gguf  # real llama.cpp (pip install 'air-sdk[llama]')

The simulated mode exercises the complete loop (analyse -> plan -> act -> reward -> learn ->
persist -> restore) on any machine. Its numbers come from a cost model, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from air_sdk import AdaptiveRuntime, GPUInfo, HardwareProfile, QLearningParams, SafetyPolicy
from air_sdk._validate import MiB
from air_sdk.backends.simulated import SimulatedBackend, SimulatedHardware
from air_sdk.testing import SyntheticModel, write_gguf

PROMPT = "Explain how QUIC works " * 8


def _show(title: str, data: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, default=str))


def _run(sdk: AdaptiveRuntime, model_path: Path, requests: int, *, discover: bool,
         n_ctx: int) -> None:
    model = sdk.load_model(model_path, discover=discover)
    _show("Model Intelligence Profile", model.analyze().as_dict())
    print("variants:", [v.quant.name for v in model.variants])
    plan = model.optimize(n_ctx=n_ctx)
    _show(f"Execution plan (source={plan.source})", plan.as_dict())
    for i in range(1, requests + 1):
        r = model.generate(PROMPT, max_tokens=64)
        if i % max(1, requests // 10) == 0 or r.rolled_back:
            c = r.config
            print(f"[{i:3d}] {r.action:<20} explored={r.explored!s:<5} "
                  f"rollback={r.rolled_back!s:<5} {c.precision:<6} kv={c.kv_type.value:<5} "
                  f"gpu_layers={c.n_gpu_layers:<2} threads={c.n_threads:<2} batch={c.n_batch:<3} "
                  f"tps={r.tokens_per_second:7.2f} reward="
                  f"{'-' if r.reward is None else f'{r.reward:+.3f}'}")
    report = model.learning_report()
    top = report.get("top_configs")
    if isinstance(top, list):  # expand the canonical JSON config keys for readability
        report["top_configs"] = [
            {**t, "config": json.loads(t["config"])}
            if isinstance(t, dict) and isinstance(t.get("config"), str) else t
            for t in top
        ]
    _show("Learning report", report)
    perf = model.performance_report()
    perf.pop("recent", None)
    _show("Performance report", perf)


def simulated_demo(requests: int) -> None:
    # A scaled-down machine and model (~3 MB of synthetic GGUF in total), deleted on exit.
    with tempfile.TemporaryDirectory(prefix="air_sdk_demo_") as tmp:
        work = Path(tmp)
        for ftype, tag in ((1, "f16"), (7, "q8_0"), (2, "q4_0")):
            write_gguf(work / f"demo-{tag}.gguf", SyntheticModel(
                name="demo-8L", n_layers=8, n_embd=128, n_head=4, n_head_kv=2, n_ff=384,
                n_vocab=1024, file_type=ftype))

        ram, vram = 64 * MiB, 3 * MiB
        hw = HardwareProfile("SimCPU-4C/8T", 4, 8, ram, ram - 32 * MiB, "Simulated",
                             (GPUInfo(0, "SimGPU", vram, vram, (8, 6)),))
        sim = SimulatedHardware(ram_bytes=ram, background_ram_bytes=32 * MiB, vram_bytes=vram,
                                physical_cores=4, logical_cores=8, cpu_bandwidth_gbs=0.02,
                                gpu_bandwidth_gbs=0.25, gpu_context_bytes=512 * 1024,
                                runtime_bytes=2 * MiB)
        safety = SafetyPolicy(ram_headroom_min=2 * MiB, vram_headroom_min=256 * 1024,
                              reserve_gpu_context_bytes=512 * 1024,
                              reserve_runtime_bytes=2 * MiB, max_quality_loss=0.06)

        def runtime() -> AdaptiveRuntime:
            return AdaptiveRuntime(
                "balanced", rl_enabled=True, auto_adapt=True,
                backend=SimulatedBackend(sim, seed=42), data_dir=work / "db", safety=safety,
                hardware=hw, seed=42, rl_params=QLearningParams(epsilon_start=0.4),
            )

        with runtime() as sdk:
            _show("Hardware Capability Profile (simulated)", sdk.hardware_profile().as_dict())
            _run(sdk, work / "demo-f16.gguf", requests, discover=True, n_ctx=1024)

        print("\n\n######## Restart: policy restored from the experience store ########")
        with runtime() as sdk:
            _run(sdk, work / "demo-f16.gguf", max(10, requests // 4), discover=True,
                 n_ctx=1024)


def real_demo(model: Path, requests: int) -> None:
    with AdaptiveRuntime("balanced", backend="llama_cpp", adapt_every=5) as sdk:
        _show("Hardware Capability Profile", sdk.hardware_profile().as_dict())
        _run(sdk, model, requests, discover=True, n_ctx=4096)


def main() -> None:
    parser = argparse.ArgumentParser(description="AIR-SDK demo")
    parser.add_argument("--model", type=Path, help="GGUF model (uses llama.cpp backend)")
    parser.add_argument("--requests", type=int, default=60)
    args = parser.parse_args()
    if args.model is not None:
        real_demo(args.model, args.requests)
    else:
        simulated_demo(args.requests)


if __name__ == "__main__":
    main()
