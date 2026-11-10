"""Benchmark every chat-capable Ollama model through the full AIR-SDK loop.

    python -m benchmarks.run_models [--episodes 12] [--models a,b] [--ctx 2048]

For each model: analyse -> optimise -> N chat episodes (RL on) -> report.
Records per episode: decode tok/s, TTFT, config, action (explore/exploit), reward,
rollback, wall time, and SDK overhead (wall time not spent inside the backend).
Writes benchmarks/results.json and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from air_sdk import AdaptiveRuntime, AirError, InsufficientResourcesError

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Explain what a CPU cache is in two sentences.",
    "Write a haiku about rain.",
    "What is 17 * 3? Reply with just the number.",
    "List three uses of Python.",
    "Translate 'good morning' into Spanish.",
    "Summarise what HTTP is in one sentence.",
    "Name two planets larger than Earth.",
]
OUT = Path(__file__).parent / "results.json"
think_s = 2.0  # simulated user think time between messages in background mode


def _col(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(v) for r in rows if isinstance(v := r.get(key), (int, float))]


def _median(rows: list[dict[str, object]], key: str, digits: int) -> float | None:
    values = _col(rows, key)
    return round(statistics.median(values), digits) if values else None


def bench_model(sdk: AdaptiveRuntime, name: str, episodes: int, ctx: int) -> dict[str, object]:
    t0 = time.perf_counter()
    model = sdk.load_ollama_model(name, discover=True)
    analyse_s = time.perf_counter() - t0
    profile = model.analyze()
    try:
        t1 = time.perf_counter()
        plan = model.optimize(n_ctx=min(ctx, profile.context_length))
        optimise_s = time.perf_counter() - t1
    except InsufficientResourcesError as exc:
        sdk.release(model)
        return {"model": name, "status": "does_not_fit", "reason": str(exc),
                "weights_gib": round(profile.weight_bytes / 2**30, 2)}

    rows: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    for i in range(episodes):
        prompt = PROMPTS[i % len(PROMPTS)]
        history = [{"role": "user", "content": prompt}]  # independent turns
        t = time.perf_counter()
        try:
            r = model.chat(history, max_tokens=64, temperature=0.2, seed=i)
        except AirError as exc:
            rows.append({"episode": i + 1, "error": str(exc)})
            continue
        wall = time.perf_counter() - t
        c = r.config
        rows.append({
            "episode": i + 1, "tps": round(r.tokens_per_second, 1), "ttft_s": round(r.ttft_s, 3),
            "total_s": round(r.total_s, 2), "wall_s": round(wall, 2),
            "overhead_s": round(wall - r.total_s, 3), "action": r.action, "explored": r.explored,
            "reward": None if r.reward is None else round(r.reward, 4),
            "rolled_back": r.rolled_back, "precision": c.precision, "threads": c.n_threads,
            "batch": c.n_batch, "gpu_layers": c.n_gpu_layers,
            "ram_mib": (r.peak_ram_bytes or 0) >> 20, "answer": r.text[:120],
        })
    learning = model.learning_report()
    ok = [r for r in rows if "tps" in r]
    rewards = [float(r["reward"]) for r in ok if r.get("reward") is not None]  # type: ignore[arg-type]
    half = len(rewards) // 2
    summary = {
        "model": name,
        "status": "ok",
        "architecture": profile.architecture,
        "params_b": round(profile.n_params / 1e9, 2),
        "variants": [v.quant.name for v in model.variants],
        "analyse_s": round(analyse_s, 2),
        "optimise_s": round(optimise_s, 2),
        "initial_plan": plan.config.as_dict() | {"source": plan.source},
        "episodes_ok": len(ok),
        "errors": len(rows) - len(ok),
        "rollbacks": sum(1 for r in ok if r["rolled_back"]),
        "explored": sum(1 for r in ok if r["explored"]),
        "config_changes": sum(1 for r in ok if r["action"] != "MAINTAIN"),
        "median_tps": _median(ok, "tps", 1),
        "median_ttft_s": _median(ok, "ttft_s", 3),
        "median_overhead_s": _median(ok, "overhead_s", 3),
        "max_overhead_s": max(_col(ok, "overhead_s"), default=None),
        "ram_mib": max(_col(ok, "ram_mib"), default=None),
        "reward_first_half": round(statistics.mean(rewards[:half]), 4) if half else None,
        "reward_second_half": round(statistics.mean(rewards[half:]), 4) if rewards[half:] else None,
        "final_config": learning.get("top_configs"),
        "episodes": rows,
    }
    sdk.release(model)
    return summary


LONG_PROMPTS = [  # each reliably produces >= 64 tokens, so speed measurements are stable
    "Explain how a CPU cache works, in about 80 words.",
    "Describe the water cycle in about 80 words.",
    "Explain what TCP does and how it differs from UDP, in about 80 words.",
    "Describe how a hash table works, in about 80 words.",
    "Explain photosynthesis in about 80 words.",
    "Describe how HTTPS keeps data private, in about 80 words.",
]


def run_mode(name: str, *, rl: bool, episodes: int, ctx: int, data_dir: Path,
             objective: str, background: bool = False) -> dict[str, object]:
    """Run ``episodes`` replies with RL on or off; return end-to-end statistics."""
    with AdaptiveRuntime(objective, rl_enabled=rl, auto_adapt=rl, backend="ollama",
                         data_dir=data_dir, seed=11, adapt_every=1,
                         background_switching=background and rl) as sdk:
        model = sdk.load_ollama_model(name, discover=True)
        plan = model.optimize(n_ctx=ctx)
        tokens, wall, reloads, reload_s, prepared = 0, 0.0, 0, 0.0, 0
        tps: list[float] = []
        configs: dict[str, int] = {}
        for i in range(episodes):
            t = time.perf_counter()
            r = model.chat([{"role": "user", "content": LONG_PROMPTS[i % len(LONG_PROMPTS)]}],
                           max_tokens=96, temperature=0.0, seed=1)
            wall += time.perf_counter() - t
            if background:
                time.sleep(think_s)  # a person reading/typing; background loads happen here
            tokens += r.completion_tokens
            tps.append(r.tokens_per_second)
            if r.switch_s > 0.01:  # ignore sub-10 ms checks of an already-finished prefetch
                reloads, reload_s = reloads + 1, reload_s + r.switch_s
            prepared += int(r.prefetched)
            c = r.config
            key = f"{c.precision}/t{c.n_threads}/b{c.n_batch}/g{c.n_gpu_layers}"
            configs[key] = configs.get(key, 0) + 1
        last = tps[len(tps) // 2:]
        return {
            "mode": ("rl+background" if background else "rl") if rl else "fixed",
            "start_config": f"{plan.config.precision}/t{plan.config.n_threads}"
                            f"/b{plan.config.n_batch}",
            "end_to_end_tok_s": round(tokens / wall, 1),
            "median_decode_tok_s": round(statistics.median(tps), 1),
            "median_decode_tok_s_2nd_half": round(statistics.median(last), 1),
            "reloads_user_waited_for": reloads,
            "reload_wait_s": round(reload_s, 2),
            "configs_prepared_in_background": prepared,
            "total_wall_s": round(wall, 1),
            "configs_used": configs,
            "learning": model.learning_report() if rl else None,
        }


def compare(name: str, episodes: int, ctx: int, root: Path, *, objective: str,
            background: bool) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out: list[dict[str, object]] = []
    for rl in (False, True):
        res = run_mode(name, rl=rl, episodes=episodes, ctx=ctx,
                       data_dir=root / f"cmp-{stamp}-{'rl' if rl else 'fixed'}",
                       objective=objective, background=background)
        brief = {k: v for k, v in res.items() if k != "learning"}
        print(json.dumps(brief, indent=1), flush=True)
        learning = res.get("learning")
        if isinstance(learning, dict):
            print(" learning:", json.dumps({k: learning[k] for k in (
                "episodes", "epsilon", "q_states", "mean_reward_early", "mean_reward_recent")}))
        out.append(res)
    (Path(__file__).parent / "compare.json").write_text(json.dumps(out, indent=1), "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--data-dir", type=Path, default=Path(".air_bench"))
    ap.add_argument("--compare", type=str, default="",
                    help="model name: run fixed-config vs RL on identical prompts")
    ap.add_argument("--objective", type=str, default="balanced")
    ap.add_argument("--background", action="store_true",
                    help="RL mode uses background switching (with a 2 s think time)")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare, args.episodes, args.ctx, args.data_dir,
                objective=args.objective, background=args.background)
        return

    results: list[dict[str, object]] = []
    with AdaptiveRuntime("balanced", backend="ollama", data_dir=args.data_dir, seed=7) as sdk:
        names = [m.name for m in sdk.list_ollama_models()]
        if args.models:
            wanted = set(args.models.split(","))
            names = [n for n in names if n in wanted]
        print(f"hardware: {json.dumps(sdk.hardware_profile().as_dict())}")
        for name in names:
            print(f"\n=== {name} ===", flush=True)
            try:
                res = bench_model(sdk, name, args.episodes, args.ctx)
            except AirError as exc:
                res = {"model": name, "status": "error", "reason": str(exc)}
            results.append(res)
            brief = {k: v for k, v in res.items() if k not in ("episodes", "final_config")}
            print(json.dumps(brief, indent=1), flush=True)
            for row in res.get("episodes", []):  # type: ignore[union-attr]
                print("  ", json.dumps(row), flush=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
