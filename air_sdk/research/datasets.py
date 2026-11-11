"""Built-in, deterministic test sets (200+ items per task) and public-benchmark importers.

Built-in sets are generated from templates with a fixed seed, so every run of the study
sees exactly the same items, and every gold answer is *computed* (SQL gold results come from
executing the gold query; maths answers from integer arithmetic), never typed by hand.

* ``sql``  - questions over a generated shop database (SQLite file written next to the items)
* ``math`` - multi-step word problems with integer answers
* ``json`` - field extraction from generated sentences
* ``code`` - small Python functions with assert tests (dev set; use MBPP for the study)

Public benchmarks (optional, downloaded on request): GSM8K test split (maths, MIT) and
MBPP (code, CC BY 4.0). Spider (SQL) must be downloaded manually; see the README.
"""

# The gold SQL below is built only from the fixed constants in this module (cities,
# categories, years...), never from user input, so string-built queries are safe here.
# ruff: noqa: S608

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import httpx

from ..errors import ValidationError
from .items import Item, write_items

SEED: Final[int] = 20261112
CITIES: Final = ("Paris", "London", "Mumbai", "Tokyo", "Berlin", "Toronto")
CATEGORIES: Final = ("books", "games", "garden", "kitchen", "sports")
STATUSES: Final = ("paid", "shipped", "cancelled")
YEARS: Final = (2023, 2024, 2025)

SHOP_SCHEMA: Final[str] = "\n".join((
    "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, joined TEXT);",
    "CREATE TABLE products (id INTEGER PRIMARY KEY, title TEXT, category TEXT, price REAL);",
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES "
    "customers(id), amount REAL, created TEXT, status TEXT);",
    "CREATE TABLE order_items (order_id INTEGER REFERENCES orders(id), product_id INTEGER "
    "REFERENCES products(id), quantity INTEGER);",
))


# ---- shop database ---------------------------------------------------------------------------


def build_shop_db(path: Path, seed: int = SEED) -> Path:
    """Write the deterministic shop database used by the SQL items."""
    rng = random.Random(seed)
    tmp = path.with_suffix(".partial")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    try:
        db.executescript(SHOP_SCHEMA)
        first = ("Asha", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jia",
                 "Kofi", "Lena", "Mo", "Nia", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tara")
        for cid in range(1, 61):
            db.execute("INSERT INTO customers VALUES (?,?,?,?)", (
                cid, f"{first[(cid - 1) % 20]} {chr(65 + (cid - 1) // 20)}{cid}",
                CITIES[rng.randrange(len(CITIES))],
                f"{YEARS[rng.randrange(3)]}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"))
        for pid in range(1, 41):
            db.execute("INSERT INTO products VALUES (?,?,?,?)", (
                pid, f"{CATEGORIES[(pid - 1) % 5]} item {pid}", CATEGORIES[(pid - 1) % 5],
                round(rng.uniform(3, 120), 2)))
        for oid in range(1, 401):
            db.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (
                oid, rng.randint(1, 60), round(rng.uniform(5, 400), 2),
                f"{YEARS[rng.randrange(3)]}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                STATUSES[rng.randrange(3)]))
            for pid in rng.sample(range(1, 41), rng.randint(1, 3)):
                db.execute("INSERT INTO order_items VALUES (?,?,?)",
                           (oid, pid, rng.randint(1, 5)))
        db.commit()
    finally:
        db.close()
    tmp.replace(path)
    return path


_JOIN: Final[str] = "FROM orders o JOIN customers c ON c.id = o.customer_id"
_ITEMS: Final[str] = ("FROM order_items i JOIN products p ON p.id = i.product_id "
                      "JOIN orders o ON o.id = i.order_id")


def _sql_templates() -> Iterator[tuple[str, str]]:
    """(question, gold SQL) pairs; every combination is unique."""
    yield from _sql_by_city()
    yield from _sql_other()


def _sql_by_city() -> Iterator[tuple[str, str]]:
    join = _JOIN
    for city in CITIES:
        yield (f"How many customers live in {city}?",
               f"SELECT COUNT(*) FROM customers WHERE city = '{city}'")
        yield (f"What is the total order amount for customers in {city}?",
               f"SELECT ROUND(SUM(o.amount), 2) {join} WHERE c.city = '{city}'")
        yield (f"What is the largest single order amount from a customer in {city}?",
               f"SELECT MAX(o.amount) {join} WHERE c.city = '{city}'")
        yield (f"How many orders were placed by customers in {city}?",
               f"SELECT COUNT(*) {join} WHERE c.city = '{city}'")
        for status in STATUSES:
            yield (f"List the names of customers in {city} who have at least one order with "
                   f"status '{status}'.",
                   f"SELECT DISTINCT c.name {join} WHERE c.city = '{city}' "
                   f"AND o.status = '{status}'")
            yield (f"How many '{status}' orders were placed by customers in {city}?",
                   f"SELECT COUNT(*) {join} WHERE c.city = '{city}' AND o.status = '{status}'")
        for year in YEARS:
            yield (f"List the names of customers from {city} who joined in {year}.",
                   f"SELECT name FROM customers WHERE city = '{city}' "
                   f"AND strftime('%Y', joined) = '{year}'")
        for k in (3, 5, 7):
            yield (f"List the names of customers in {city} who placed more than {k} orders.",
                   f"SELECT c.name {join} WHERE c.city = '{city}' GROUP BY c.id "
                   f"HAVING COUNT(*) > {k}")


def _sql_other() -> Iterator[tuple[str, str]]:
    join, items = _JOIN, _ITEMS
    for x in (50, 100, 150, 200, 250, 300, 350):
        yield (f"How many orders have an amount greater than {x}?",
               f"SELECT COUNT(*) FROM orders WHERE amount > {x}")
    for status in STATUSES:
        yield (f"What is the average amount of orders with status '{status}'?",
               f"SELECT ROUND(AVG(amount), 2) FROM orders WHERE status = '{status}'")
        for x in (25, 75, 125, 175, 225):
            yield (f"How many orders with status '{status}' have an amount below {x}?",
                   f"SELECT COUNT(*) FROM orders WHERE status = '{status}' AND amount < {x}")
    for cat in CATEGORIES:
        yield (f"What is the average price of products in the {cat} category?",
               f"SELECT ROUND(AVG(price), 2) FROM products WHERE category = '{cat}'")
        yield (f"What is the total quantity sold of products in the {cat} category?",
               f"SELECT SUM(i.quantity) {items} WHERE p.category = '{cat}'")
        yield (f"How many distinct customers ordered products from the {cat} category?",
               f"SELECT COUNT(DISTINCT o.customer_id) {items} WHERE p.category = '{cat}'")
        for year in YEARS:
            yield (f"What is the total revenue (quantity times product price) of the {cat} "
                   f"category for orders created in {year}?",
                   f"SELECT ROUND(SUM(i.quantity * p.price), 2) {items} "
                   f"WHERE p.category = '{cat}' AND strftime('%Y', o.created) = '{year}'")
        for lo, hi in ((5, 30), (30, 60), (60, 90), (90, 120)):
            yield (f"List the titles of {cat} products priced between {lo} and {hi}.",
                   f"SELECT title FROM products WHERE category = '{cat}' "
                   f"AND price BETWEEN {lo} AND {hi}")
    for p in (10, 20, 40, 60, 80, 100):
        yield (f"List the titles of products cheaper than {p}.",
               f"SELECT title FROM products WHERE price < {p}")
    for year in YEARS:
        yield (f"How many orders were placed in {year}?",
               f"SELECT COUNT(*) FROM orders WHERE strftime('%Y', created) = '{year}'")
        yield (f"How many customers joined in {year}?",
               f"SELECT COUNT(*) FROM customers WHERE strftime('%Y', joined) = '{year}'")
        for q, months in enumerate((("01", "03"), ("04", "06"), ("07", "09"), ("10", "12")), 1):
            yield (f"What is the total order amount in quarter {q} of {year}?",
                   f"SELECT ROUND(SUM(amount), 2) FROM orders WHERE created BETWEEN "
                   f"'{year}-{months[0]}-01' AND '{year}-{months[1]}-31'")
    for n in (1, 2, 3, 5):
        yield (f"Which {n} customer(s) have the highest total order amount? Return names, "
               f"highest first.",
               f"SELECT c.name {join} GROUP BY c.id ORDER BY SUM(o.amount) DESC LIMIT {n}")
    yield ("Which city has the most customers?",
           "SELECT city FROM customers GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1")
    yield ("What is the total order amount per city? Return city and total.",
           f"SELECT c.city, ROUND(SUM(o.amount), 2) {join} GROUP BY c.city")
    yield ("How many products are in each category? Return category and count.",
           "SELECT category, COUNT(*) FROM products GROUP BY category")


def sql_items(db_name: str = "shop") -> list[Item]:
    return [Item(f"sql-{i:03d}", "sql", q, {"query": gold, "db": db_name}, SHOP_SCHEMA)
            for i, (q, gold) in enumerate(_sql_templates(), 1)]


# ---- maths ----------------------------------------------------------------------------------


def _math_templates(rng: random.Random) -> Iterator[tuple[str, int]]:
    names = ("Asha", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jia")
    for _ in range(30):
        a, b, c = rng.randint(3, 40), rng.randint(4, 25), rng.randint(1, 30)
        n = names[rng.randrange(10)]
        yield (f"{n} buys {a} boxes with {b} pencils in each box and gives away {c} pencils. "
               f"How many pencils does {n} have left?", a * b - c)
        price, qty, paid = rng.randint(5, 60), rng.randint(2, 9), 0
        paid = price * qty + rng.randint(1, 200)
        yield (f"A notebook costs {price} rupees. {n} buys {qty} notebooks and pays with "
               f"{paid} rupees. How much change does {n} get?", paid - price * qty)
        k = rng.randint(2, 9)
        total = k * rng.randint(5, 80)
        yield (f"{total} sweets are shared equally among {k} children. How many sweets does "
               f"each child get?", total // k)
        speed, hours, extra = rng.randint(30, 90), rng.randint(2, 8), rng.randint(5, 60)
        yield (f"A train travels at {speed} km per hour for {hours} hours, then another "
               f"{extra} km. How far does it travel in total, in km?", speed * hours + extra)
        p, base = rng.choice((10, 20, 25, 50)), rng.randint(2, 40) * 20
        yield (f"A shirt costs {base} rupees and is sold at a {p}% discount. What is the sale "
               f"price in rupees?", base - base * p // 100)
        x, y, z = rng.randint(10, 99), rng.randint(10, 99), rng.randint(10, 99)
        s = x + y + z
        while s % 3:
            z += 1
            s = x + y + z
        yield (f"Three tests have scores {x}, {y} and {z}. What is the average score?", s // 3)
        w, length = rng.randint(3, 30), rng.randint(3, 30)
        yield (f"A garden is {w} metres wide and {length} metres long. A path goes all the way "
               f"around its edge. How many metres long is the path?", 2 * (w + length))
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        yield (f"A class has {a} rows of desks with {b} desks in each row. {a} desks are "
               f"broken. How many desks can be used?", a * b - a)


def math_items() -> list[Item]:
    rng = random.Random(SEED + 1)
    out: list[Item] = []
    seen: set[str] = set()
    for q, ans in _math_templates(rng):
        if q in seen:
            continue
        seen.add(q)
        out.append(Item(f"math-{len(out) + 1:03d}", "math", q, {"answer": str(ans)}))
    return out


# ---- JSON extraction -------------------------------------------------------------------------


def json_items() -> list[Item]:
    rng = random.Random(SEED + 2)
    names = ("Asha Rao", "Ben Ortiz", "Chen Wei", "Dara Khan", "Eli Moss", "Fatima Noor",
             "Gus Hall", "Hana Sato", "Ivan Petrov", "Jia Lin")
    out: list[Item] = []
    for i in range(1, 241):
        kind = i % 3
        name, city = names[rng.randrange(10)], CITIES[rng.randrange(len(CITIES))]
        if kind == 0:
            oid, amount = f"ORD-{rng.randint(1000, 9999)}", rng.randint(10, 5000)
            date = f"{YEARS[rng.randrange(3)]}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            fields: dict[str, object] = {"order_id": oid, "customer": name, "city": city,
                                         "date": date, "amount": amount}
            text = rng.choice((
                f"Order {oid} was placed by {name} from {city} on {date} for {amount} rupees.",
                f"On {date}, {name} ({city}) placed order {oid}, paying {amount} rupees.",
            ))
        elif kind == 1:
            age = rng.randint(18, 80)
            user = name.split()[0].lower()
            email = f"{user}{rng.randint(1, 99)}@example.com"
            fields = {"name": name, "age": age, "email": email, "city": city}
            text = rng.choice((
                f"{name} is {age} years old, lives in {city} and can be reached at {email}.",
                f"Contact: {email}. {name}, aged {age}, is based in {city}.",
            ))
        else:
            title = f"{CATEGORIES[rng.randrange(5)]} kit {rng.randint(1, 99)}"
            price, stock = rng.randint(2, 900), rng.randint(0, 500)
            fields = {"title": title, "price": price, "stock": stock}
            text = rng.choice((
                f"The product '{title}' costs {price} dollars and {stock} units are in stock.",
                f"In stock: {stock} units of '{title}', priced at {price} dollars each.",
            ))
        keys = ", ".join(fields)
        out.append(Item(f"json-{i:03d}", "json",
                        f"Extract a JSON object with the keys {keys} from this text: {text}",
                        {"fields": fields}, keys))
    return out


# ---- code (dev set) --------------------------------------------------------------------------

_CODE: Final[tuple[tuple[str, str, str], ...]] = (
    ("sum_list", "returns the sum of a list of numbers",
     "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"),
    ("max_of", "returns the largest number in a non-empty list",
     "assert max_of([3, 9, 2]) == 9\nassert max_of([-5, -2]) == -2"),
    ("reverse_string", "returns the string reversed",
     "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''"),
    ("is_palindrome", "returns True if a string reads the same backwards, ignoring case",
     "assert is_palindrome('Level')\nassert not is_palindrome('hello')"),
    ("count_vowels", "returns the number of vowels (a, e, i, o, u) in a string, any case",
     "assert count_vowels('Hello') == 2\nassert count_vowels('xyz') == 0"),
    ("factorial", "returns n! for a non-negative integer n",
     "assert factorial(0) == 1\nassert factorial(5) == 120"),
    ("fibonacci", "returns the n-th Fibonacci number with fibonacci(0) == 0 and fibonacci(1) == 1",
     "assert fibonacci(0) == 0\nassert fibonacci(10) == 55"),
    ("is_prime", "returns True if an integer n is prime",
     "assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\nassert not is_prime(9)"),
    ("gcd", "returns the greatest common divisor of two positive integers",
     "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1"),
    ("celsius_to_fahrenheit", "converts a temperature from Celsius to Fahrenheit",
     "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212"),
    ("flatten", "flattens a list of lists into one list",
     "assert flatten([[1, 2], [3], []]) == [1, 2, 3]"),
    ("unique_sorted", "returns the unique values of a list in ascending order",
     "assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]"),
    ("word_count", "returns a dict mapping each word in a string to how often it appears",
     "assert word_count('a b a') == {'a': 2, 'b': 1}"),
    ("second_largest", "returns the second largest distinct number in a list",
     "assert second_largest([4, 1, 4, 3]) == 3"),
    ("rotate_left", "rotates a list left by k positions",
     "assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]\n"
     "assert rotate_left([1, 2], 0) == [1, 2]"),
    ("chunk", "splits a list into consecutive chunks of size n",
     "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"),
    ("count_words", "returns the number of words in a sentence separated by spaces",
     "assert count_words('the cat sat') == 3\nassert count_words('') == 0"),
    ("capitalize_words", "capitalizes the first letter of every word in a string",
     "assert capitalize_words('hello world') == 'Hello World'"),
    ("digit_sum", "returns the sum of the digits of a non-negative integer",
     "assert digit_sum(1234) == 10\nassert digit_sum(0) == 0"),
    ("is_anagram", "returns True if two strings are anagrams of each other",
     "assert is_anagram('listen', 'silent')\nassert not is_anagram('a', 'b')"),
    ("merge_sorted", "merges two sorted lists into one sorted list",
     "assert merge_sorted([1, 4], [2, 3]) == [1, 2, 3, 4]"),
    ("binary_search", "returns the index of a target in a sorted list, or -1 if absent",
     "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3], 4) == -1"),
    ("remove_duplicates", "removes duplicates from a list, keeping first-occurrence order",
     "assert remove_duplicates([3, 1, 3, 2, 1]) == [3, 1, 2]"),
    ("average", "returns the average of a non-empty list of numbers",
     "assert average([2, 4, 6]) == 4"),
    ("power_set_size", "returns how many subsets a set of n elements has",
     "assert power_set_size(3) == 8\nassert power_set_size(0) == 1"),
    ("to_binary", "returns the binary representation of a non-negative integer as a string",
     "assert to_binary(5) == '101'\nassert to_binary(0) == '0'"),
    ("count_char", "returns how many times a character appears in a string",
     "assert count_char('banana', 'a') == 3"),
    ("list_product", "returns the product of a list of numbers (1 for an empty list)",
     "assert list_product([2, 3, 4]) == 24\nassert list_product([]) == 1"),
    ("even_numbers", "returns the even numbers of a list, in order",
     "assert even_numbers([1, 2, 3, 4]) == [2, 4]"),
    ("max_key", "returns the key with the largest value in a non-empty dict",
     "assert max_key({'a': 1, 'b': 5}) == 'b'"),
)


def code_items() -> list[Item]:
    return [Item(f"code-{i:03d}", "code",
                 f"Write a Python function `{name}` that {what}.",
                 {"tests": tests, "entry": name})
            for i, (name, what, tests) in enumerate(_CODE, 1)]


BUILDERS: Final[dict[str, Callable[[], list[Item]]]] = {
    "sql": sql_items, "math": math_items, "json": json_items, "code": code_items,
}


def write_builtin(out_dir: Path) -> dict[str, int]:
    """Write all built-in sets (and the shop database) into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    build_shop_db(out_dir / "shop.sqlite")
    counts: dict[str, int] = {}
    for task, build in BUILDERS.items():
        meta: dict[str, object] = {"task": task, "source": "EdgeMind built-in generator",
                                   "seed": SEED}
        if task == "code":
            meta["note"] = "small dev set; use MBPP for the study"
        counts[task] = write_items(out_dir / f"{task}.jsonl", build(), meta)
    return counts


# ---- public benchmarks -------------------------------------------------------------------------

PUBLIC: Final[dict[str, tuple[str, str, str]]] = {
    # name: (url, task, licence)
    "gsm8k": ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
              "grade_school_math/data/test.jsonl", "math", "MIT"),
    "mbpp": ("https://raw.githubusercontent.com/google-research/google-research/master/"
             "mbpp/mbpp.jsonl", "code", "CC BY 4.0"),
}
_MAX_DOWNLOAD: Final[int] = 20 << 20
_DEF: Final = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(", re.M)


def _download(url: str) -> str:
    with httpx.Client(timeout=60.0, follow_redirects=True) as client, \
            client.stream("GET", url) as resp:
        resp.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        for chunk in resp.iter_bytes():
            size += len(chunk)
            if size > _MAX_DOWNLOAD:
                raise ValidationError(f"{url} is larger than {_MAX_DOWNLOAD >> 20} MiB")
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


def convert_public(name: str, text: str) -> list[Item]:
    """Turn a downloaded benchmark file into items."""
    out: list[Item] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if name == "gsm8k":
            answer = str(row["answer"]).rsplit("####", 1)[-1].strip().replace(",", "")
            out.append(Item(f"gsm8k-{n:04d}", "math", str(row["question"]), {"answer": answer}))
        elif name == "mbpp":
            m = _DEF.search(str(row["code"]))
            if not m:
                continue
            tests = "\n".join(str(t) for t in row["test_list"])
            out.append(Item(f"mbpp-{row['task_id']}", "code",
                            f"{row['text']} The function must be named `{m.group(1)}`.",
                            {"tests": tests, "entry": m.group(1)}))
    return out


def fetch_public(name: str, out_dir: Path) -> int:
    if name not in PUBLIC:
        raise ValidationError(f"unknown benchmark {name!r}; choose from {sorted(PUBLIC)}")
    url, task, licence = PUBLIC[name]
    items = convert_public(name, _download(url))
    return write_items(out_dir / f"{name}.jsonl", items,
                       {"task": task, "source": url, "license": licence})
