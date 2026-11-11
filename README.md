# EdgeMind — run any LLM, automatically optimised for your machine

**Repository:** <https://github.com/sripad2020/EdgeMind> · **Overview page:** `site/index.html`

EdgeMind loads a language model, studies your hardware, and picks the fastest setup that
safely fits — then keeps improving it while you use it. You write three lines; it handles
precision, CPU/GPU split, memory, context size, recovery from errors and learning.

> The Python package is `air_sdk` and the command is `air-sdk` (the project's original
> working name was AIR-SDK, the Adaptive Intelligence Runtime).

```python
import air_sdk

llm = air_sdk.load("qwen2.5:0.5b")
print(llm.ask("What is the capital of France?"))
```

## What it solves

| Problem | What EdgeMind does |
|---|---|
| The model does not fit, freezes the machine, or runs slowly with default settings | Reads the model and free memory first, refuses unsafe setups, compresses the model, and tunes itself on that exact machine (tabular Q-learning) |
| One compression level is used for every task: fast but error-prone, or accurate but slow | Answers on the fastest version and moves to a more accurate one only when the task needs it |
| A model invents a column or a total, and the answer looks just as confident | Checks SQL, code, JSON and numbers with fixed rules; each answer is passed, retried on a stronger version, or marked failed |
| Sensitive data may not leave the company's systems | Everything runs on the device; nothing is sent to a cloud service |
| Long conversations get slow and mix topics together | One memory section per task; only relevant facts are sent, within a token budget |
| Nobody knows what a compressed model can really do on this hardware | Pass/fail evidence is recorded per task and per version on the device itself |

## Why it matters

- **Measured, not claimed.** What a model can do is measured on the device it runs on, at
  each compression level, not taken from leaderboards run on large GPUs.
- **Proof with every answer.** Checkable answers carry their result (checked against the
  schema, the expected number, or the required fields) instead of only looking confident.
- **No second AI to trust.** Every decision uses fixed rules plus one small Q-learning
  table, so each choice can be explained and audited.
- **Hardware companies already own.** Ordinary laptops and office PCs run the model, which
  removes per-request cloud fees.

## Edge AI in fintech

Banks, lenders, payment companies and accounting firms hold data that often may not leave
their own systems. EdgeMind runs the model on machines they already operate and labels
every answer that feeds money, records or reports.

| Use | Check that applies |
|---|---|
| KYC / onboarding: fields from application text | JSON parses, required fields present |
| Analyst questions on internal data ("loans disbursed per branch this quarter") | SQL compiles against the real schema, one read-only statement |
| Invoices and reconciliation | Fields present; totals recomputed by the safe calculator *(roadmap)* |
| Transaction categorisation | Answer must be one of the allowed labels *(roadmap)* |
| EMI, interest and ratio explanations | Final number matches the calculator |
| Policy and compliance questions | No automatic check: labelled "unchecked" for human review |

Why on the edge: data stays on the premises, every answer has an audit record, routing is
explainable, branches keep working offline, and cost does not grow with request volume.
EdgeMind helps with data residency and auditability; it is not a regulatory certification.

**Research.** The study *Verified Execution Profiles for Edge LLMs* (section 9) measures,
per task, the cheapest adapter + compression combination that is statistically no worse
than the full model.

---

## 1. Install (once)

1. **Python 3.10+**
2. **Ollama** — download from <https://ollama.com> and start it (open the app, or run `ollama serve`).
3. **EdgeMind** — straight from GitHub:
   ```bash
   pip install "edgemind[web] @ git+https://github.com/sripad2020/EdgeMind"
   ```
   or from a downloaded copy of the repository:
   ```bash
   git clone https://github.com/sripad2020/EdgeMind
   cd EdgeMind
   pip install -e ".[web]"     # [web] adds the chat website; leave it out for the SDK only
   ```
4. **Check your setup** — it tells you exactly what to fix:
   ```bash
   air-sdk doctor              # or: python -m air_sdk doctor
   ```

## 2. Use any model

`air_sdk.load(...)` accepts **any** of these — nothing else changes in your code:

| You have… | Write | Example |
|---|---|---|
| A model installed in Ollama | its name | `air_sdk.load("llama3.2:1b")` |
| A model from the Ollama library, not yet downloaded | name + `pull=True` | `air_sdk.load("phi3:mini", pull=True)` |
| Any GGUF model on HuggingFace | `hf.co/<user>/<repo>[:<quant>]` or the URL | `air_sdk.load("hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M", pull=True)` |
| A `.gguf` file on your disk | its path | `air_sdk.load("C:/models/mistral-7b.Q4_K_M.gguf")` |

This covers practically every open model (Llama, Qwen, Gemma, Mistral, Phi, DeepSeek,
SmolLM, …) because they are all distributed as GGUF. Closed cloud models (GPT, Claude, …)
cannot be used: they do not run on your hardware, so there is nothing to optimise.

## 3. The whole API on one screen

```python
import air_sdk

llm = air_sdk.load("qwen2.5:0.5b")              # load + optimise for this machine

llm.ask("question")                              # -> answer text
llm.reply("question")                            # -> Reply (text + speed + what was chosen)
for piece in llm.stream("question"):             # -> answer word by word
    print(piece, end="")

chat = llm.conversation(system="Be brief.")      # sectioned memory (see section 5)
chat.say("Hi, I'm Asha.")
chat.say("What's my name?")                      # -> "Asha"
chat.reset()

llm.summary()                                    # plain-English explanation of the setup
llm.report()                                     # everything as a dict
llm.make_faster()                                # create Q8/Q4 versions + measure quality
llm.measure_quality()                            # how much each version changes answers
llm.close()                                      # free memory, save what was learned

air_sdk.list_models()                            # installed models + "fits in memory now?"
air_sdk.doctor()                                 # setup checklist
```

`with air_sdk.load(...) as llm:` closes it automatically.

### Options of `load()` (all optional)

| Option | Default | Meaning |
|---|---|---|
| `device` | `"auto"` | `"auto"`, `"cpu"`, or `"gpu"` (NVIDIA) |
| `goal` | `"balanced"` | `"balanced"`, `"speed"`, `"latency"` (fast first word), `"memory"` |
| `learn` | `True` | keep improving the setup with reinforcement learning |
| `max_context` | `4096` | longest conversation; memory is only reserved when it is needed |
| `speed_test` | `True` | measure each available precision on load, start on the best |
| `pull` | `False` | download the model if it is missing |
| `quality_limit` | `0.05` | how much answer change compressed versions may cause (0–1) |
| `reserve_ram_gb` | 10% of RAM | memory always left free for your other programs |
| `data_dir` | `~/.air_sdk` | where learning is saved |

## 4. Verified skills — the speed of Q4, the correctness of Q8

Tell EdgeMind what the task is. For tasks that can be checked automatically, it answers on
the **fastest** precision, **checks** the answer with a fixed rule (no AI judge, nothing is
executed), and **retries on a more accurate precision only if the check fails**. Every
result is remembered per model, task and precision, so the next answer starts where it
is known to pass.

```python
schema = "CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL);"
r = llm.reply("Total amount per customer", skill="sql", context=schema)
r.verified, r.escalated, r.attempts      # True, False, (("Q4_K_M", True, "SQL is valid ..."),)

llm.use("python")                        # default task for following questions
llm.skill_report("sql")                  # passes/fails per precision, where it starts next
```

| Skill | Check | `context` |
|---|---|---|
| `sql` | Query is compiled (`EXPLAIN`) against your schema in a private in-memory database: tables and columns must exist; never executed; one statement only | schema (`CREATE TABLE …`) |
| `python` | Code must parse (`ast.parse`); never executed | — |
| `json` | Must parse; must contain the required keys | `name, age` or an example object |
| `math` | Final number must match the expected value | expected number |
| `general`, `summarise` | No automatic check (answered once) | — |

**How blame is assigned:** a precision is marked as failing a task only when a more
accurate precision then *passed* the same request. If every precision fails, only the most
accurate attempt is counted: the task is beyond the model, not the fast precision's fault.

Measured live on this laptop (`qwen2.5:0.5b`, Q4_K_M + Q8_0, 4 SQL questions with a real
schema and 3 maths questions): 6 of 7 answers passed **on the fast Q4_K_M**; one SQL
question failed on both precisions and was reported as failed instead of being returned
as if it were correct.

Command line: `air-sdk ask qwen2.5:0.5b "Average order per city" --skill sql --context
"CREATE TABLE …"` (exit code 3 if the answer fails its check), `air-sdk skills <model>`,
and `/skill sql` inside `air-sdk chat`. The website has a **Task** picker beside the chat box.

You usually do not need to name the task: the default `skill="auto"` detects it per
message (see the next section).

## 5. Context-aware conversations — sectioned memory

A long chat moves between pieces of work: a database, then a Python helper, a quick
calculation, back to the database. Sending the whole history every time makes prompts
long and slow, and mixes topics up. EdgeMind keeps **sections** instead:

```python
chat = llm.conversation()
chat.say(f"My database:\n{schema}\nTotal order amount per customer name.")  # SQL section
chat.say("now only customers in Paris")                 # follow-up: same section
chat.say("Write a Python function that slugifies a title")  # new section (Python)
chat.say("add type hints to it")                        # stays in the Python section
chat.say("What is 17 * 23?")                            # maths: checked against 391
chat.say("Back to the orders query: sort by total")     # returns to the SQL section
chat.say("New task: summarise this text ...")           # explicit new section

r = chat.reply("...")
r.label      # 'Section: Tables: customers, orders (sql) - task sql (detected) - check passed'
chat.sections()   # what is remembered: preferences + each section's facts
```

**What happens for every message (rules only, no extra model):**

1. **Facts are extracted** and stored in the section: SQL schemas (CREATE / ALTER / DROP
   applied in order, so the newest definition wins), definitions ("X means Y"), Python
   function names, required JSON keys, the expected result of a calculation (computed by a
   safe parser, never `eval`), and global preferences ("always …").
2. **The task is detected** (SQL, Python, JSON, maths, summarise, general). A weak signal
   such as "now sort it" keeps the section's current task.
3. **The message is routed** to a section by shared names (tables, functions, terms),
   keywords and recency; "back to …" and "new task: …" are understood. A clearly different
   task starts a new section even if a few everyday words overlap.
4. **A small prompt is built** within a token budget: only the active section; only the
   tables the question names plus the tables linked to them by a foreign key, in compact
   form (`orders(id PK, customer_id->customers, amount)`); then as many recent turns as
   fit. If it still does not fit: drop less relevant facts → drop older turns → (the
   context grows automatically) → last resort, the bare question with a notice.
5. **The answer is checked against the full stored facts**, not the trimmed prompt: a
   query that uses a table left out of the prompt still passes, and one that invents a
   table fails. General answers that contain SQL, Python or JSON get a bonus check.
6. **Corrections** ("no, that's wrong …") after an answer that passed its check count as a
   failure for that precision, so the next answer starts on a more accurate one.

Limits keep memory bounded: 8 sections (least recently used removed first), 24 turns per
section, 64 tables, 32 definitions/names, 8 preferences.

Measured live (`qwen2.5:0.5b`, 7-message conversation across SQL / Python / maths / JSON):
each message was routed to the right section; all 7 answers passed their checks; prompts
were 7–353 estimated tokens (the Python question carried no schema at all: ~23 tokens).

**Servers:** `air_sdk.context.SessionStore` keeps one sectioned memory per conversation id
(30-minute idle expiry, bounded count, a lock per conversation). The website sends a random
conversation id, so the server holds the memory and the browser sends only the new message.

## 6. Command line

```bash
air-sdk doctor                            # check setup
air-sdk models                            # installed models, and which fit in memory now
air-sdk ask   qwen2.5:0.5b "What is QUIC?"
air-sdk chat  qwen2.5:0.5b                # interactive (/sections /summary /reset /bye)
air-sdk pull  hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M
air-sdk import C:/models/my-model.gguf    # use a local file
air-sdk faster smollm2:360m               # make Q8/Q4 versions + measure quality
air-sdk quality smollm2:360m
air-sdk web --port 8000                   # chat website
```

## 7. Chat website

```bash
air-sdk web          # then open http://127.0.0.1:8000
```

Pick a model (or **Add any model…** by name / HuggingFace link), device and goal, click
**Load & optimise**, and chat. Side panels show the plan, live RL decisions, learning,
speed tests, memory/context adaptation and measured quality. Local-only and hardened
(see *Security* below).

## 8. Examples

| File | Shows |
|---|---|
| `examples/01_quickstart.py` | doctor, load, ask, summary |
| `examples/02_streaming_and_chat.py` | streaming, conversations, reply details |
| `examples/03_any_model.py` | Ollama library, HuggingFace, local files, friendly errors |
| `examples/04_make_it_faster.py` | create faster versions and measure their quality |
| `examples/05_all_options.py` | every option + the advanced engine |
| `examples/06_verified_skills.py` | verified skills: SQL / maths / JSON / Python checks and escalation |
| `examples/07_sectioned_context.py` | one conversation across several tasks: sections, routing, budget |

## 9. Research harness — verified execution profiles

`python -m air_sdk.research` runs the study *Verified Execution Profiles for Edge LLMs*:
which adapter + precision each task needs on this device, with statistical proof.

```bash
E=experiments/main
python -m air_sdk.research --exp $E init                      # built-in sets: SQL 202, maths 231, JSON 240, code 30 (dev)
python -m air_sdk.research --exp $E fetch gsm8k               # optional public benchmarks (gsm8k, mbpp)
python -m air_sdk.research --exp $E build smollm2:360m --baseline --uniform --sensitivity 8   # RQ1 builds
python -m air_sdk.research --exp $E profile smollm2:360m smollm2:360m-air-q8_0 smollm2:360m-air-q4_0 \
    smollm2:360m-air-sens8-g0 smollm2:360m-air-sens8-g1 smollm2:360m-air-sens8-g2 smollm2:360m-air-sens8-g3 \
    --files sql,math,json --n-ctx 1024
python -m air_sdk.research --exp $E report                    # report.md + report.json
python -m air_sdk.research --exp $E suggest-mix smollm2:360m --task sql --keep 2   # -> e.g. sql:1,3
python -m air_sdk.research --exp $E build smollm2:360m --mix sql:1,3 --group-size 8  # RQ2 build
python -m air_sdk.research --exp $E build smollm2:360m --adapter math.gguf --label math --on smollm2:360m-air-q4_0  # RQ3
python -m air_sdk.research --exp $E clean smollm2:360m        # remove this study's builds from Ollama
```

| Part | What it does |
|---|---|
| Items | JSONL per task, split 80% profile / 20% router by a hash of the id (stable) |
| Online check | What EdgeMind sees: SQL compiles, number present, JSON keys, Python parses |
| Gold score | Evaluation only: SQL execution accuracy on a read-only database, exact number, every JSON field, code tests in a locked-down Docker container (no network, read-only, non-root, 5 s). Without Docker, code gets no gold score — never a plain subprocess |
| Builds | Uniform Q8/Q4, one layer group at Q4 (sensitivity), task-aware mixes, LoRA adapters (GGUF) — all from the F16 source by EdgeMind's quantizer; base weights untouched |
| Grid | Wilson 95% intervals; a profile is *qualified* only if its lower bound beats the base model |
| Policies (RQ4/5) | Exact replay of logs: fixed profiles, cheapest-first cascade, tuned threshold, tabular Q-learning; tuned on the profile split, reported on the router split |
| Report | Grid, sensitivity map with McNemar tests, adapter gains, policies, checks vs gold |

Needs an F16 source model (e.g. `smollm2:360m`); runs are resumable; temperature 0, fixed seed.

## 10. When something goes wrong

Every error says what happened **and how to fix it**, e.g.:

```
Model 'smollm2' is not installed in Ollama.
  -> How to fix: Did you mean: smollm2:360m? Otherwise download it with `ollama pull smollm2`,
     or call air_sdk.load('smollm2', pull=True).
```

| Message | Fix |
|---|---|
| Cannot reach Ollama | Start Ollama (open the app or `ollama serve`) |
| Model … is not installed | `pull=True`, `air-sdk pull <name>`, or use a suggested name |
| no configuration fits the currently free memory | close programs, use a smaller / Q4 model (the message lists models that fit now) |
| … cannot chat | it is an embedding model; pick a chat model (`air-sdk models`) |
| device='gpu' but no GPU | use `device="auto"`; GPU mode needs an NVIDIA GPU |

---

## What happens automatically (how it works)

```
Import → Analyze → Optimize → Execute → Observe → Reward → Learn → Improve
```

Four **deterministic** engines analyse the model and hardware and produce *validated*
configurations; a **tabular Q-learning** engine — the only learning component — chooses
among them and improves from measured results. No neural networks, no supervised ML, no
external AI services; the model's weights are never changed.

| Feature | What it does |
|---|---|
| **Model analysis** | Reads any GGUF header safely (all architectures; tolerant of missing metadata and attention-free models) |
| **Safe memory planning** | Refuses setups that would not fit; estimates calibrated against measured usage |
| **Smart start** | Predicts the fastest setup from the hardware, or measures it with a quick speed test |
| **Automatic quantization** | Turns F16/BF16/F32 models into Q8_0 / Q4_0 (byte-exact vs llama.cpp's reference) |
| **Measured quality** | 12 fixed questions; word agreement with the original replaces built-in estimates |
| **Background switching** | Next configuration is prepared while you read, so changes cost no waiting |
| **Memory-pressure relief** | If other programs take memory, switches to a safe setup before the PC swaps |
| **Automatic context** | Starts at 1024 tokens, grows only when the conversation needs it |
| **Rollback** | Out-of-memory / load failures roll back to a working setup; you still get the answer |
| **Persistence** | What works best on *this* machine is remembered across restarts |

### Measured on a CPU-only laptop (Intel i5 4C/8T, 16 GB, Ollama 0.40)

`smollm2:360m` + SDK-created Q8_0 / Q4_0:

| | Plain default (F16, fixed) | AIR-SDK |
|---|---|---|
| Speed from the first message | 44 tok/s | **111 tok/s** (speed test picks Q4_0) |
| 30 replies end-to-end (learning) | 43.5 tok/s | **68.2 tok/s (+57%)**, 0 s waiting for reloads |
| Answer agreement with F16 | — | Q8_0 85%, Q4_0 57% (measured; Q4 is faster but changes more) |

Reproduce: `python -m benchmarks.run_models --compare smollm2:360m --episodes 30 --objective throughput --background`

### Advanced API

`air_sdk.load()` is a thin layer over the full engine, which stays available:

```python
from air_sdk import AdaptiveRuntime
with AdaptiveRuntime("throughput", backend="ollama", background_switching=True) as rt:
    model = rt.load_ollama_model("qwen2.5:0.5b")          # or rt.load_model("x.gguf") w/ llama.cpp
    plan = model.optimize(device_placement="auto", n_ctx=4096, probe=True)
    r = model.chat([{"role": "user", "content": "hi"}], on_token=print)
    print(model.policy_snapshot(), model.learning_report())
```

Backends: `ollama` (default), `llama_cpp` (`pip install -e ".[llama]"`), `simulated` (tests).

### Layout

| Module | Role |
|---|---|
| `easy.py`, `cli.py` | Simple API (`load`, `LLM`, `Conversation`, `doctor`) and command line |
| `skills/` | Verified skills: deterministic answer checks, verify-then-escalate, pass/fail routing |
| `context/` | Sectioned memory: fact extraction, task detection, routing, token budget, safe calculator, server session store |
| `analysis/` | Hardened GGUF parser, Model Intelligence Profile, quantization catalogue |
| `hardware/` | Hardware profile (psutil, NVML) and bounded background telemetry |
| `engines/` | Precision candidates, memory estimation / CPU-GPU placement, candidate lattice |
| `planner.py` | Budgets, safety masking, prediction, relief, conservative fallback |
| `rl/` | State, actions, reward, Q-table, SQLite persistence, agent |
| `quantize.py`, `quality.py` | Deterministic quantizer, measured answer quality |
| `backends/` | Ollama, llama.cpp, simulated |
| `runtime.py` | Episode loop, background switching, rollback, reports |
| `webapp/` | FastAPI chat website |

### RL design

* **State**: discretised free RAM/VRAM, CPU/GPU load, memory pressure, model size, request
  length, temperature + current configuration.
* **Actions (11)**: keep; precision up/down; more/fewer GPU layers; more/fewer threads;
  larger/smaller batch; KV-cache precision up/down.
* **Safety**: only actions whose target passes the deterministic checks (live memory, quality
  limit, failure blacklist) are offered — exploration is safe by construction.
* **Reward**: `R = w1·T + w2·L + w3·M − w4·Q − w5·S − P` (speed, latency, memory, quality,
  waiting-for-reload share, failure penalty), weights per goal.
* **Update**: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`; unvisited entries start from
  the measured value of the configuration they lead to.

### Security & engineering

* Strict validation at every boundary (no bool-as-int, finite floats, bounded strings, checked paths and model names).
* GGUF parser bounds every count/length by hard caps and remaining file size; tensor data is never read during analysis.
* Parameterised SQL only, validated rows, no pickle; owner-only database on POSIX.
* Website: localhost-only, DNS-rebinding protection, JSON-only POSTs with Origin check, strict CSP, model output rendered as text.
* `mypy --strict` and ruff (incl. bandit) clean; 110+ tests.

### Known limitations

* Configuration changes reload the model (hidden by background switching in chat; batch workloads should use `adapt_every`).
* Quantization targets are Q8_0 / Q4_0 from F16/BF16/F32 sources only (no K-quants yet).
* Ollama's KV-cache type is server-wide, so it is not changed per request.
* GPU support is NVIDIA-only (NVML); other GPUs are treated as absent. GPU decisions are not yet benchmarked on real NVIDIA hardware.
* Ollama 0.40 cannot use models tagged exactly `:local`; imports use `:air-local`.
