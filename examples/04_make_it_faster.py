"""4. Make a model faster on this machine (and check what it costs in quality).

Works for full-precision (F16/BF16/F32) models, e.g. `ollama pull smollm2:360m`.
Run:  python examples/04_make_it_faster.py
"""

import air_sdk

with air_sdk.load("smollm2:360m", goal="speed") as llm:
    print("Before:\n" + llm.summary())

    # Creates Q8_0 and Q4_0 versions, measures how much their answers change compared
    # with the original, and reloads so AIR-SDK can switch between them.
    result = llm.make_faster()
    print("Created:", result["created"])
    quality = result.get("quality")
    if isinstance(quality, dict):
        for row in quality.get("results", []):
            print(f"  {row['model']:30} {round(100 * row['agreement'])}% same words as the "
                  f"original, quality loss {row['measured_loss']:.3f}")

    print(llm.ask("Name three planets."))
    print("After:\n" + llm.summary())
