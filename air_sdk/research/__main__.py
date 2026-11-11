"""Command line for the study: ``python -m air_sdk.research <command> --exp DIR``.

    init                       write the built-in test sets + shop database
    fetch gsm8k|mbpp           download a public benchmark into the item folder
    build BASE --baseline --uniform --sensitivity 4 --mix sql:1,3 --adapter F --label L --on V
    suggest-mix BASE --task sql --keep 2    the most sensitive groups (from RQ1 results)
    profile MODEL...           answer + check + gold-score items on each model
    rescore                    re-run gold scoring on stored answers (e.g. after Docker)
    report                     write report.md / report.json
    clean BASE                 remove this study's builds of BASE from Ollama
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..backends.ollama import DEFAULT_BASE_URL, OllamaClient
from ..errors import AirError
from ..quantize import QuantTarget
from .build import Builder
from .datasets import PUBLIC, fetch_public, write_builtin
from .gold import docker_available
from .items import Item, read_items, select
from .report import build_report
from .runner import EdgeMindGenerator, profile_items, rescore
from .store import Profile, ResultStore

DEFAULT_FILES = ("sql", "math", "json", "code")


def _store(exp: Path) -> ResultStore:
    return ResultStore(exp / "results.sqlite")


def _items(exp: Path, files: Sequence[str]) -> list[Item]:
    out: list[Item] = []
    for name in files:
        path = exp / "items" / f"{name}.jsonl"
        if not path.is_file():
            raise AirError(f"{path} not found; run `init` (or `fetch {name}`) first")
        out += read_items(path)[1]
    return out


def cmd_init(a: argparse.Namespace) -> int:
    counts = write_builtin(a.exp / "items")
    _store(a.exp).close()
    print(json.dumps(counts))
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    n = fetch_public(a.name, a.exp / "items")
    url, _task, licence = PUBLIC[a.name]
    print(f"{n} items from {url} ({licence})")
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    store = _store(a.exp)
    b = Builder(OllamaClient(a.ollama_url), a.exp / "work")
    made: list[Profile] = []
    if a.baseline:
        made.append(b.baseline(a.base))
    if a.uniform:
        made += [b.uniform(a.base, QuantTarget.Q8_0), b.uniform(a.base, QuantTarget.Q4_0)]
    if a.sensitivity:
        made += b.sensitivity(a.base, a.sensitivity)
    for spec in a.mix or []:
        label, _, groups = spec.partition(":")
        keep = {int(g) for g in groups.split(",") if g.strip()}
        made.append(b.mix(a.base, keep, a.group_size, label))
    if a.adapter:
        made.append(b.adapter(a.on or a.base, Path(a.adapter), a.label, a.base))
    for p in made:
        store.add_profile(p)
        print(f"{p.kind:12} {p.name}  ({p.size_bytes / 2**20:.0f} MB)")
    store.close()
    return 0


def cmd_suggest(a: argparse.Namespace) -> int:
    store = _store(a.exp)
    report = json.loads((a.exp / "report.json").read_text(encoding="utf-8")) \
        if (a.exp / "report.json").is_file() else None
    store.close()
    if not report:
        raise AirError("run `report` after profiling the sensitivity builds first")
    rows = [r for r in report["sensitivity"] if r["base"] == a.base and r["task"] == a.task]
    if not rows:
        raise AirError(f"no sensitivity results for {a.base} / {a.task}")
    worst = sorted(rows, key=lambda r: -float(r["drop"]))[: a.keep]
    groups = sorted(int(r["group"]) for r in worst)
    print(f"{a.task}:{','.join(map(str, groups))}")
    return 0


def cmd_profile(a: argparse.Namespace) -> int:
    items = _items(a.exp, a.files)
    if a.tasks:
        wanted = set(a.tasks.split(","))
        items = [i for i in items if i.task in wanted]
    split = None if a.split == "all" else a.split
    items = [i for task in sorted({i.task for i in items})  # --limit is per task
             for i in select((x for x in items if x.task == task), split=split, limit=a.limit)]
    sandbox = docker_available()
    if any(i.task == "code" for i in items) and not sandbox:
        print("note: Docker sandbox not available - code answers get no gold score "
              "(run `rescore` later)", file=sys.stderr)
    store = _store(a.exp)
    gen = EdgeMindGenerator(data_dir=a.exp / "runtime", reserve_ram_gb=a.reserve_ram_gb,
                            n_ctx=a.n_ctx, ollama_url=a.ollama_url)
    try:
        for model in a.models:
            if store.profile(model) is None:
                store.add_profile(Profile(model, model, "model"))

            def show(n: int, total: int, r: object, model: str = model) -> None:
                if n % 10 == 0 or n == total:
                    print(f"  {model}: {n}/{total}", file=sys.stderr)

            ran = profile_items(store, gen, model, items, data_dir=a.exp / "items",
                                sandbox=sandbox, max_tokens=a.max_tokens, on_result=show)
            print(f"{model}: {ran} new answers")
    finally:
        gen.close()
        store.close()
    return 0


def cmd_rescore(a: argparse.Namespace) -> int:
    store = _store(a.exp)
    n = rescore(store, _items(a.exp, a.files), data_dir=a.exp / "items",
                sandbox=docker_available())
    store.close()
    print(f"{n} gold scores changed")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    store = _store(a.exp)
    path = build_report(store, a.exp)
    store.close()
    print(path)
    return 0


def cmd_clean(a: argparse.Namespace) -> int:
    store = _store(a.exp)
    b = Builder(OllamaClient(a.ollama_url), a.exp / "work")
    installed = set(b.client.list_models())
    for p in store.profiles():
        if p.base == a.base and p.kind in ("sensitivity", "mix", "adapter") \
                and p.name in installed:
            b.remove(p.name)
            print(f"removed {p.name}")
    store.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m air_sdk.research")
    ap.add_argument("--exp", type=Path, default=Path("experiments/main"))
    ap.add_argument("--ollama-url", default=DEFAULT_BASE_URL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    f = sub.add_parser("fetch")
    f.add_argument("name", choices=sorted(PUBLIC))
    f.set_defaults(fn=cmd_fetch)
    b = sub.add_parser("build")
    b.add_argument("base")
    b.add_argument("--baseline", action="store_true")
    b.add_argument("--uniform", action="store_true")
    b.add_argument("--sensitivity", type=int, metavar="GROUP_SIZE")
    b.add_argument("--mix", action="append", metavar="LABEL:G1,G2")
    b.add_argument("--group-size", type=int, default=4)
    b.add_argument("--adapter")
    b.add_argument("--label", default="adapter")
    b.add_argument("--on", help="attach the adapter to this variant (default: the base)")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("suggest-mix")
    s.add_argument("base")
    s.add_argument("--task", required=True)
    s.add_argument("--keep", type=int, default=2)
    s.set_defaults(fn=cmd_suggest)
    p = sub.add_parser("profile")
    p.add_argument("models", nargs="+")
    p.add_argument("--files", type=lambda v: v.split(","), default=list(DEFAULT_FILES))
    p.add_argument("--tasks")
    p.add_argument("--split", choices=["all", "profile", "router"], default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--reserve-ram-gb", type=float, default=0.5)
    p.add_argument("--n-ctx", type=int, default=2048,
                   help="context size (prompts here are short; 1024 saves memory)")
    p.set_defaults(fn=cmd_profile)
    r = sub.add_parser("rescore")
    r.add_argument("--files", type=lambda v: v.split(","), default=list(DEFAULT_FILES))
    r.set_defaults(fn=cmd_rescore)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    c = sub.add_parser("clean")
    c.add_argument("base")
    c.set_defaults(fn=cmd_clean)
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except AirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
