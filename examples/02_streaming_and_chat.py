"""2. Streaming answers and multi-turn conversations.

Run:  python examples/02_streaming_and_chat.py
"""

import air_sdk

with air_sdk.load("qwen2.5:0.5b") as llm:
    # Stream: print the answer word by word as it is generated.
    for piece in llm.stream("Write a two-line poem about the sea."):
        print(piece, end="", flush=True)
    print()

    # Every reply carries what AIR-SDK did to produce it.
    r = llm.last
    if r is not None:
        print(f"[{r.tokens_per_second:.0f} tokens/s, first word after "
              f"{r.time_to_first_token_s:.2f}s, {r.precision} on {r.device}]")

    # A conversation remembers earlier messages.
    chat = llm.conversation(system="You are a friendly teacher. Keep answers short.")
    print(chat.say("My name is Asha. What is a CPU?"))
    print(chat.say("What is my name, and how is a GPU different?"))

    # Stream inside a conversation too.
    for piece in chat.stream("Give me one example of each."):
        print(piece, end="", flush=True)
    print()
