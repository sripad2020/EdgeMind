"""1. Quick start: load a model and ask a question.

Run:  python examples/01_quickstart.py
Needs: Ollama running, and one chat model installed (e.g. `ollama pull qwen2.5:0.5b`).
"""

import air_sdk

# Check that everything is set up (prints a checklist with fixes for anything missing).
print(air_sdk.doctor())

# Load a model. AIR-SDK analyses it and your hardware, tests the available precisions,
# and picks the fastest setup that fits in memory - automatically.
with air_sdk.load("qwen2.5:0.5b") as llm:
    answer = llm.ask("What is the capital of France? Answer in one sentence.")
    print(answer)

    # What did AIR-SDK choose, and why?
    print(llm.summary())
