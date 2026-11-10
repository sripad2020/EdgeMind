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

chat = llm.conversation(system="Be brief.")      # remembers earlier messages
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

## 4. Command line

```bash
air-sdk doctor                            # check setup
air-sdk models                            # installed models, and which fit in memory now
air-sdk ask   qwen2.5:0.5b "What is QUIC?"
air-sdk chat  qwen2.5:0.5b                # interactive (/summary /reset /bye)
air-sdk pull  hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M
air-sdk import C:/models/my-model.gguf    # use a local file
air-sdk faster smollm2:360m               # make Q8/Q4 versions + measure quality
air-sdk quality smollm2:360m
air-sdk web --port 8000                   # chat website
```

## 5. Chat website

```bash
air-sdk web          # then open http://127.0.0.1:8000
```

Pick a model (or **Add any model…** by name / HuggingFace link), device and goal, click
**Load & optimise**, and chat. Side panels show the plan, live RL decisions, learning,
speed tests, memory/context adaptation and measured quality. Local-only and hardened
(see *Security* below).

## 6. Examples

| File | Shows |
|---|---|
| `examples/01_quickstart.py` | doctor, load, ask, summary |
| `examples/02_streaming_and_chat.py` | streaming, conversations, reply details |
| `examples/03_any_model.py` | Ollama library, HuggingFace, local files, friendly errors |
| `examples/04_make_it_faster.py` | create faster versions and measure their quality |
| `examples/05_all_options.py` | every option + the advanced engine |

## 7. When something goes wrong

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
