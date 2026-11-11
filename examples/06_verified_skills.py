"""6. Verified skills: the speed of the small precision, the correctness of the large one.

Tell EdgeMind the task. It answers on the fastest precision, checks the answer with a
deterministic rule, and retries on a more accurate precision only if the check fails.

Run:  python examples/06_verified_skills.py
Works best with a model that has more than one precision installed, e.g. qwen2.5:0.5b
(Q4_K_M) plus qwen2.5:0.5b-instruct-q8_0, or after `air-sdk faster smollm2:360m`.
"""

import air_sdk

SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
"""

with air_sdk.load("qwen2.5:0.5b") as llm:
    # SQL: the query is compiled against the schema (never run). Wrong table/column names
    # fail the check and trigger a retry on a more accurate precision.
    r = llm.reply("Total order amount per customer name.", skill="sql", context=SCHEMA)
    print(r.text)
    print(f"verified={r.verified}  escalated={r.escalated}  path={[a[0] for a in r.attempts]}"
          f"  ({r.check})")

    # Maths with a known answer: the final number must match.
    r = llm.reply("What is 17 * 23?", skill="math", context="391")
    print(f"maths: verified={r.verified} on {r.precision} ({r.check})")

    # JSON with required keys.
    r = llm.reply("Give a JSON object describing a book.", skill="json",
                  context="title, author, year")
    print(f"json: verified={r.verified} ({r.check})")

    # Make a skill the default for following questions (and conversations).
    llm.use("python")
    print(llm.ask("Write a function that reverses a string."))
    print("python check:", llm.last.check if llm.last else "-")

    # Evidence per skill: passes/fails per precision, and where the next answer starts.
    for skill in ("sql", "math", "json", "python"):
        report = llm.skill_report(skill)
        print(skill, "starts on", report["starts_on"], "|", report["precisions"])
