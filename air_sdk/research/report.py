"""Turn an experiment's results into the tables for RQ1-RQ6 (Markdown + JSON)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..quantize import QuantTarget, derived_model_name
from .controller import compare
from .stats import mcnemar_exact
from .store import Cell, Profile, ResultStore

SELECTABLE = ("model", "uniform", "mix", "adapter")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _paired(store: ResultStore, a: str, b: str, task: str) -> tuple[list[bool], list[bool]]:
    ra = {r.item_id: r for r in store.results(profile=a, task=task, split="profile")}
    rb = {r.item_id: r for r in store.results(profile=b, task=task, split="profile")}
    ids = sorted(i for i in ra.keys() & rb.keys()
                 if ra[i].gold_ok is not None and rb[i].gold_ok is not None)
    return [bool(ra[i].gold_ok) for i in ids], [bool(rb[i].gold_ok) for i in ids]


def grid_section(cells: list[Cell], profiles: dict[str, Profile]) -> list[str]:
    lines = ["## Grid (profile split)", "",
             "| Task | Profile | Kind | Size MB | Check pass | Gold accuracy [95% CI] | "
             "Mean s | s per correct | Qualified |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in sorted(cells, key=lambda c: (c.task, c.profile)):
        p = profiles.get(c.profile)
        lines.append(
            f"| {c.task} | {c.profile} | {p.kind if p else '?'} | "
            f"{(p.size_bytes / 2**20) if p else 0:.0f} | {_pct(c.check.value)} | "
            f"{_pct(c.gold.value)} [{_pct(c.gold.low)}, {_pct(c.gold.high)}] (n={c.gold.n}) | "
            f"{c.mean_seconds:.2f} | "
            f"{'-' if c.seconds_per_correct is None else f'{c.seconds_per_correct:.2f}'} | "
            f"{'yes' if c.qualified else ('baseline' if p and p.baseline else 'no')} |")
    return lines


def sensitivity_section(store: ResultStore, cells: list[Cell],
                        profiles: dict[str, Profile]) -> tuple[list[str], list[dict[str, object]]]:
    """RQ1: gold accuracy drop vs uniform Q8_0 when one layer group is quantized to Q4_0."""
    rows: list[dict[str, object]] = []
    by = {(c.profile, c.task): c for c in cells}
    for p in profiles.values():
        if p.kind != "sensitivity":
            continue
        ref = derived_model_name(p.base, QuantTarget.Q8_0)
        for task in sorted({c.task for c in cells}):
            cell, ref_cell = by.get((p.name, task)), by.get((ref, task))
            if cell is None or ref_cell is None:
                continue
            a, b = _paired(store, ref, p.name, task)
            only_ref, only_sens, pval = mcnemar_exact(a, b)
            rows.append({"base": p.base, "group": int(p.groups), "task": task,
                         "q8_accuracy": ref_cell.gold.value, "accuracy": cell.gold.value,
                         "drop": ref_cell.gold.value - cell.gold.value,
                         "lost": only_ref, "gained": only_sens, "p_value": pval})
    if not rows:
        return [], rows
    lines = ["## RQ1 - sensitivity map (accuracy drop vs uniform Q8_0)", "",
             "| Base | Task | " + " | ".join(
                 f"g{g}" for g in sorted({int(str(r['group'])) for r in rows})) + " |"]
    groups = sorted({int(str(r["group"])) for r in rows})
    lines.append("| --- | --- | " + " | ".join("---" for _ in groups) + " |")
    grouped: dict[tuple[str, str], dict[int, dict[str, object]]] = defaultdict(dict)
    for r in rows:
        grouped[(str(r["base"]), str(r["task"]))][int(str(r["group"]))] = r
    for (base_name, task_name), cols in sorted(grouped.items()):
        cells_txt = []
        for g in groups:
            entry = cols.get(g)
            if entry is None:
                cells_txt.append("-")
                continue
            drop = float(str(entry["drop"]))
            star = "*" if float(str(entry["p_value"])) < 0.05 else ""
            cells_txt.append(f"{100 * drop:+.1f}{star}")
        lines.append(f"| {base_name} | {task_name} | " + " | ".join(cells_txt) + " |")
    lines += ["", "Positive = accuracy lost. * = McNemar p < 0.05 vs uniform Q8_0 "
              "(paired, same items)."]
    return lines, rows


def adapter_section(cells: list[Cell], profiles: dict[str, Profile]) -> list[str]:
    """RQ3: adapter gain over the model it is attached to, per precision of that model."""
    by = {(c.profile, c.task): c for c in cells}
    lines: list[str] = []
    for p in profiles.values():
        if p.kind != "adapter":
            continue
        on = str(json.loads(p.meta).get("on", ""))
        for (name, task), cell in sorted(by.items()):
            if name != p.name or (on, task) not in by:
                continue
            ref = by[(on, task)]
            lines.append(f"| {p.adapter} | {on} ({p.precision}) | {task} | "
                         f"{_pct(ref.gold.value)} | {_pct(cell.gold.value)} | "
                         f"{100 * (cell.gold.value - ref.gold.value):+.1f} |")
    if not lines:
        return []
    return ["## RQ3 - adapter gain by base precision", "",
            "| Adapter | Attached to | Task | Without | With | Gain (points) |",
            "| --- | --- | --- | --- | --- | --- |", *lines]


def checks_section(store: ResultStore) -> tuple[list[str], dict[str, dict[str, int]]]:
    """RQ6: online check vs gold, per task, over every answer."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in store.results():
        if r.gold_ok is None:
            continue
        key = ("pass" if r.check_ok else "fail") + "/" + ("correct" if r.gold_ok else "wrong")
        counts[r.task][key] += 1
    lines = ["## RQ6 - online check vs gold", "",
             "| Task | Answers | Check pass & correct | Check pass but wrong | "
             "Check fail but correct | Check fail & wrong | Wrong among passed |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for task, c in sorted(counts.items()):
        n = sum(c.values())
        passed = c["pass/correct"] + c["pass/wrong"]
        lines.append(f"| {task} | {n} | {c['pass/correct']} | {c['pass/wrong']} | "
                     f"{c['fail/correct']} | {c['fail/wrong']} | "
                     f"{_pct(c['pass/wrong'] / passed) if passed else '-'} |")
    return lines, {t: dict(v) for t, v in counts.items()}


def build_report(store: ResultStore, out_dir: Path) -> Path:
    profiles = {p.name: p for p in store.profiles()}
    cells = store.grid("profile")
    lines = ["# EdgeMind study report", ""]
    lines += [*grid_section(cells, profiles), ""]
    sens_lines, sens_rows = sensitivity_section(store, cells, profiles)
    lines += [*sens_lines, ""]
    lines += [*adapter_section(cells, profiles), ""]
    policies: list[dict[str, object]] = []
    tasks = sorted({c.task for c in cells})
    pol_lines = ["## RQ4 / RQ5 - selection policies (router split, never tuned on)", "",
                 "| Task | Policy | n | Gold accuracy [95% CI] | Mean s | s per correct | "
                 "Profiles tried |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for base in sorted({p.base for p in profiles.values()}):
        names = [p.name for p in profiles.values() if p.base == base and p.kind in SELECTABLE]
        for task in tasks:
            for rep in compare(store, task, names):
                policies.append({"base": base, **rep.as_dict()})
                spc = "-" if rep.seconds_per_correct is None \
                    else f"{rep.seconds_per_correct:.2f}"
                pol_lines.append(
                    f"| {task} | {rep.policy} | {rep.n} | {_pct(rep.accuracy)} "
                    f"[{_pct(rep.ci_low)}, {_pct(rep.ci_high)}] | {rep.mean_seconds:.2f} | "
                    f"{spc} | {rep.mean_tried:.2f} |")
    if policies:
        lines += [*pol_lines, ""]
    check_lines, check_counts = checks_section(store)
    lines += check_lines
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({
        "grid": [c.as_dict() for c in cells], "sensitivity": sens_rows,
        "policies": policies, "checks_vs_gold": check_counts,
    }, indent=2), encoding="utf-8")
    return md
