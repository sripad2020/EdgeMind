"""5. Every option of air_sdk.load(), and access to the full engine.

Run:  python examples/05_all_options.py
"""

import json

import air_sdk

llm = air_sdk.load(
    "qwen2.5:0.5b",
    device="auto",          # "auto" (recommended) | "cpu" | "gpu" (NVIDIA)
    goal="balanced",        # "balanced" | "speed" | "latency" | "memory"
    learn=True,             # keep improving the setup with reinforcement learning
    max_context=8192,       # longest conversation supported (memory is reserved only when needed)
    speed_test=True,        # measure each available precision on load, start on the best
    pull=False,             # download the model if it is missing
    quality_limit=0.05,     # max acceptable quality loss for compressed versions (0..1)
    reserve_ram_gb=None,    # RAM always left free for other programs (default 10% of RAM)
    data_dir=None,          # where learning is saved (default ~/.air_sdk)
)
try:
    reply = llm.reply("Explain what a cache is in one sentence.", max_tokens=80,
                      temperature=0.2, system="Be precise.")
    print(reply.text)
    print(f"speed={reply.tokens_per_second:.1f} tok/s  precision={reply.precision}  "
          f"device={reply.device}  context={reply.context_tokens}  action={reply.action}  "
          f"explored={reply.explored}  reward={reply.reward}")

    # Full report (model profile, plan, performance, learning) as a dict:
    print(json.dumps(llm.report()["performance"], indent=1, default=str)[:1500])

    # The full engine is still available for advanced use:
    model = llm.model              # AdaptiveModel: optimize(), policy_snapshot(), ...
    print(json.dumps(model.policy_snapshot()["actions"][:3], indent=1))
finally:
    llm.close()
