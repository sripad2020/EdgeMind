"""AIR-SDK command line.

    air-sdk doctor                         check your setup
    air-sdk models                         list installed models (and which fit in memory now)
    air-sdk ask   <model> "question"       one question, streamed answer
                  [--skill sql --context "CREATE TABLE ..."]   checked answer, auto-escalation
    air-sdk chat  <model>                  chat with sectioned memory (/sections, /skill, /bye)
    air-sdk skills <model>                 skills and their pass/fail evidence
    air-sdk pull  <model>                  download a model (Ollama name or hf.co/...)
    air-sdk import <file.gguf> [--name n]  use a local GGUF file
    air-sdk faster <model>                 create Q8_0/Q4_0 versions + measure their quality
    air-sdk quality <model>                measure answer quality of installed versions
    air-sdk web [--port 8000]              start the chat website

(``python -m air_sdk ...`` works the same way.)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .easy import LLM, doctor, list_models, load
from .errors import AirError, SetupError


def _progress(status: str, done: int | None, total: int | None) -> None:
    if done is not None and total:
        pct = 100 * done / total
        sys.stderr.write(
            f"\r  {status[:40]:40} {pct:5.1f}%  ({done / 1e9:.2f}/{total / 1e9:.2f} GB)")
    else:
        sys.stderr.write(f"\r  {status[:70]:70}")
    sys.stderr.flush()


def _add_load_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="Ollama name, hf.co/<user>/<repo>[:quant], or a .gguf file")
    p.add_argument("--goal", default="balanced",
                   choices=["balanced", "speed", "latency", "memory"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature (default: the skill's recommended value)")
    p.add_argument("--skill", default=None,
                   help="task: auto (default: detected per message), general, sql, python, "
                        "json, math, summarise (verified skills check each answer and retry "
                        "on a more accurate precision if needed)")
    p.add_argument("--context", default=None,
                   help="what the skill checks against: SQL schema, required JSON keys, or "
                        "the expected number for math")
    p.add_argument("--no-learn", action="store_true", help="turn reinforcement learning off")
    p.add_argument("--no-speed-test", action="store_true", help="skip the quick speed test")
    p.add_argument("--pull", action="store_true", help="download the model if missing")
    p.add_argument("--reserve-ram-gb", type=float, default=None,
                   help="RAM to keep free for other programs (default: 10%% of RAM)")


def _load(args: argparse.Namespace) -> LLM:
    print(f"Loading {args.model} and optimising for this machine...", file=sys.stderr)
    llm = load(args.model, goal=args.goal, device=args.device, max_context=args.max_context,
               learn=not args.no_learn, speed_test=not args.no_speed_test, pull=args.pull,
               reserve_ram_gb=args.reserve_ram_gb, on_progress=_progress)
    plan = llm.model.plan
    if plan is not None:
        c = plan.config
        print(f"Ready: {c.precision} on {'CPU' if c.n_gpu_layers == 0 else 'GPU'}, "
              f"{c.n_threads} threads (chosen by {plan.source}).\n", file=sys.stderr)
    return llm


def _check_line(llm: LLM) -> str:
    r = llm.last
    if r is None:
        return ""
    if r.verified is None:
        check = "no automatic check"
    else:
        check = f"check {'PASSED' if r.verified else 'FAILED'}: {r.check}"
        if r.escalated:
            check += " (after " + " -> ".join(a[0] for a in r.attempts) + ")"
    task = f"{r.skill}{' (detected)' if r.auto else ''}"
    section = f"Section: {r.section} | " if r.section else ""
    notice = f"\n[{r.notice}]" if r.notice else ""
    return (f"[{section}{task} | {check} | {r.tokens_per_second:.1f} tok/s, "
            f"{r.precision} on {r.device}]{notice}")


def _cmd_ask(args: argparse.Namespace) -> int:
    with _load(args) as llm:
        for piece in llm.stream(args.prompt, skill=args.skill, context=args.context,
                                max_tokens=args.max_tokens, temperature=args.temperature):
            print(piece, end="", flush=True)
        print()
        print("\n" + _check_line(llm), file=sys.stderr)
        if llm.last is not None and llm.last.verified is False:
            return 3  # answered, but the answer did not pass its check
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    with load(args.model, speed_test=False, learn=False,
              reserve_ram_gb=args.reserve_ram_gb) as llm:
        print(f"Skills for {llm.name} (precisions: {', '.join(llm.model.precision_ladder())})\n")
        for s in llm.skills():
            print(f"{s['skill']:10} {s['description']}")
            if s["verified"]:
                rows = s["precisions"]
                evidence = ", ".join(
                    f"{r['precision']} {r['passes']}/{r['passes'] + r['fails']}"
                    for r in (rows if isinstance(rows, list) else [])
                    if r["passes"] + r["fails"]) or "no results yet"
                print(f"{'':10} starts on {s['starts_on']} · passed checks: {evidence}")
                if s["context"]:
                    print(f"{'':10} context: {s['context']}")
            print()
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    with _load(args) as llm:
        if args.skill:
            llm.use(args.skill)
        chat = llm.conversation(system=args.system, context=args.context)
        print("Type your message. Commands: /summary  /sections  /reset  /skill <name|auto>"
              "  /bye\nTip: 'new task: ...' starts a fresh topic; 'back to ...' returns to one.",
              file=sys.stderr)
        while True:
            try:
                text = input("\n>>> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text in ("/bye", "/exit", "/quit"):
                break
            if text == "/summary":
                print(llm.summary())
                continue
            if text == "/reset":
                chat.reset()
                print("(conversation cleared)")
                continue
            if text == "/sections":
                print(_sections_text(chat.sections()))
                continue
            if text.startswith("/skill"):
                name = text.removeprefix("/skill").strip() or None
                try:
                    llm.use(name)
                except AirError as exc:
                    print(exc)
                    continue
                print(f"(skill: {llm.skill})")
                continue
            for piece in chat.stream(text, max_tokens=args.max_tokens,
                                     temperature=args.temperature):
                print(piece, end="", flush=True)
            print()
            print(_check_line(llm), file=sys.stderr)
    return 0


def _sections_text(snap: dict[str, object]) -> str:
    lines = []
    prefs = snap.get("preferences")
    if isinstance(prefs, list) and prefs:
        lines.append("Preferences: " + "; ".join(str(p) for p in prefs))
    sections = snap.get("sections")
    for s in sections if isinstance(sections, list) else []:
        if not isinstance(s, dict):
            continue
        lines.append(f"[{s['state']}] {s['title']} ({s['skill']}) - {s['turns']} turns")
        if s["tables"]:
            lines.append(f"    tables: {', '.join(s['tables'])} (schema v{s['schema_version']})")
        if s["definitions"]:
            lines.append(f"    terms: {', '.join(s['definitions'])}")
        if s["code_names"]:
            lines.append(f"    code: {', '.join(s['code_names'])}")
    return "\n".join(lines) or "(nothing remembered yet)"


def _cmd_doctor(_args: argparse.Namespace) -> int:
    report = doctor()
    print(report)
    return 0 if report.ok else 1


def _cmd_models(_args: argparse.Namespace) -> int:
    models = list_models()
    if not models:
        print("No models installed. Try: air-sdk pull qwen2.5:0.5b")
        return 0
    print(f"{'NAME':34} {'ARCH':11} {'PARAMS':8} {'PRECISION':9} {'SIZE':>7}  CHAT  FITS NOW")
    for m in models:
        print(f"{m['name']!s:34} {m['architecture']!s:11} {m['parameters']!s:8} "
              f"{m['precision']!s:9} {m['size_gb']!s:>5}GB  {'yes' if m['can_chat'] else 'no ':4}"
              f"  {'yes' if m['fits_now'] else 'no'}")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    from .runtime import AdaptiveRuntime  # noqa: PLC0415

    with AdaptiveRuntime(backend="ollama") as rt:
        rt.pull_model(args.model, _progress)
    print(f"\nDone: {args.model} is ready. Try: air-sdk chat {args.model}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from .runtime import AdaptiveRuntime  # noqa: PLC0415

    with AdaptiveRuntime(backend="ollama") as rt:
        name = rt.import_gguf(args.path, args.name)
    print(f"Imported as {name!r}. Try: air-sdk chat {name}")
    return 0


def _cmd_faster(args: argparse.Namespace) -> int:
    from .runtime import AdaptiveRuntime  # noqa: PLC0415

    with AdaptiveRuntime(backend="ollama") as rt:
        plan = rt.quantization_plan(args.model)
        if not plan["eligible"]:
            print(plan["note"])
            return 0
        print("Creating compressed versions (this can take a minute)...")
        for r in rt.auto_quantize(args.model):
            state = f"created in {r.seconds}s" if r.created else "already exists"
            print(f"  {r.model_name:34} {r.size_bytes / 1e9:.2f} GB  {state}")
        print("Measuring answer quality against the original...")
        _print_quality(rt.evaluate_quality(args.model))
    print(f"\nAIR-SDK will now switch between these versions: air-sdk chat {args.model}")
    return 0


def _cmd_quality(args: argparse.Namespace) -> int:
    from .runtime import AdaptiveRuntime  # noqa: PLC0415

    with AdaptiveRuntime(backend="ollama") as rt:
        _print_quality(rt.evaluate_quality(args.model))
    return 0


def _print_quality(report: dict[str, object]) -> None:
    if report.get("note"):
        print(report["note"])
        return
    print(f"  reference: {report['reference']}")
    results = report.get("results")
    for r in results if isinstance(results, list) else []:
        verdict = "ok" if r["within_limit"] else "over the limit (will not be used)"
        print(f"  {r['model']:34} {round(100 * r['agreement'])}% same words, "
              f"loss {r['measured_loss']:.4f}  {verdict}")
    skipped = report.get("skipped")
    if skipped:
        print(f"  skipped (not enough free RAM now): {skipped}")


def _cmd_web(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        raise SetupError("The website needs extra packages.",
                         hint="pip install 'air-sdk[web]'") from None
    print(f"Open http://127.0.0.1:{args.port} in your browser (Ctrl+C to stop).")
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="air-sdk", description="AIR-SDK: run any LLM, automatically optimised for "
                                    "this machine.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check your setup").set_defaults(func=_cmd_doctor)
    sub.add_parser("models", help="list installed models").set_defaults(func=_cmd_models)
    p = sub.add_parser("ask", help="ask one question")
    _add_load_options(p)
    p.add_argument("prompt")
    p.set_defaults(func=_cmd_ask)
    p = sub.add_parser("chat", help="interactive chat")
    _add_load_options(p)
    p.add_argument("--system", default=None, help="system prompt")
    p.set_defaults(func=_cmd_chat)
    p = sub.add_parser("skills", help="list skills and their pass/fail evidence for a model")
    p.add_argument("model")
    p.add_argument("--reserve-ram-gb", type=float, default=None)
    p.set_defaults(func=_cmd_skills)
    p = sub.add_parser("pull", help="download a model")
    p.add_argument("model")
    p.set_defaults(func=_cmd_pull)
    p = sub.add_parser("import", help="use a local .gguf file")
    p.add_argument("path")
    p.add_argument("--name", default=None)
    p.set_defaults(func=_cmd_import)
    p = sub.add_parser("faster", help="create faster compressed versions + measure quality")
    p.add_argument("model")
    p.set_defaults(func=_cmd_faster)
    p = sub.add_parser("quality", help="measure answer quality of installed versions")
    p.add_argument("model")
    p.set_defaults(func=_cmd_quality)
    p = sub.add_parser("web", help="start the chat website")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_web)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except SetupError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except AirError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
