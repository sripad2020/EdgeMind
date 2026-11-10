"""3. Use ANY model: Ollama library, HuggingFace, or a local GGUF file.

Run:  python examples/03_any_model.py
The first two may download several hundred MB.
"""

import air_sdk

# See what is installed and what fits in your free memory right now.
for m in air_sdk.list_models():
    print(f"{m['name']:34} {m['parameters']:>8}  {m['precision']:7}  "
          f"chat={m['can_chat']}  fits_now={m['fits_now']}")

# a) Any model from the Ollama library - downloaded automatically with pull=True.
with air_sdk.load("llama3.2:1b", pull=True) as llm:
    print(llm.ask("Say hello in French."))

# b) Any GGUF model on HuggingFace (a URL works too).
with air_sdk.load("hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M", pull=True) as llm:
    print(llm.ask("Say hello in Spanish."))

# c) Any GGUF file on your disk - it is validated, then registered with Ollama for you.
#    with air_sdk.load("C:/models/my-model.Q4_K_M.gguf") as llm:
#        print(llm.ask("Say hello in German."))

# Something wrong? Every error says how to fix it:
try:
    air_sdk.load("this-model-does-not-exist:1b")
except air_sdk.SetupError as error:
    print(error)
