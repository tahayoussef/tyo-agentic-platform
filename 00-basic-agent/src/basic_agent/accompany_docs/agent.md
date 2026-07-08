# Companion notes: `agent.py`

> A junior-friendly, concept-by-concept walkthrough of [`../agent.py`](../agent.py).
> This is where the LLM meets the tools. It explains the new ideas in that file —
> generators and `yield`, the `Any` type, what a "graph" / ReAct loop is, the facade
> pattern, `@staticmethod`, and more — with tiny examples, then points back at the code.

A few ideas (the `from __future__ import annotations` line, keyword-only `*`, and
`.get_secret_value()` on secrets) were already covered in [`tools.md`](tools.md); we won't
repeat them in full here.

---

## The shape of the file

Three top-level functions and one class:

- `build_llm(settings)` — creates the chat model (your original `ChatNVIDIA` snippet).
- `build_agent(settings)` — wires the model + tools into a runnable **ReAct graph**.
- `BasicAgent` — a small, friendly wrapper (a *facade*) with two verbs: `ask` and `stream`.

We'll take the new concepts in the order they appear.

---

## Concept 1: `logger = get_logger(__name__)` and `__name__`

[`agent.py:21`](../agent.py#L21) creates one logger for the whole module:

```python
logger = get_logger(__name__)
```

`__name__` is a built-in variable Python sets automatically for every file. When this file is
imported as part of the package, `__name__` is the string `"basic_agent.agent"`. So the
logger is *tagged* with where it lives — when you read logs later, you can see the message
came from `basic_agent.agent` rather than some other module. Creating it once at the top
(module level) means every function in the file can use the same `logger`.

(There's a second, special value: if you run a file *directly* with `python agent.py`, its
`__name__` becomes `"__main__"` instead. That's the trick behind the
`if __name__ == "__main__":` line you see at the bottom of [`../cli.py`](../cli.py#L69) — it
means "only run this when executed directly, not when imported.")

---

## Concept 2: `SYSTEM_PROMPT` — a module-level constant, and what a system prompt is

[`agent.py:23-29`](../agent.py#L23-L29) defines a big string.

**Why is it in `ALL_CAPS`?** Python has no real "constant" keyword. By convention, a variable
written in `ALL_CAPS_WITH_UNDERSCORES` signals "this is a fixed value; don't reassign it."
It's a message to other humans, not enforced by the language.

**What is a system prompt?** When you talk to a chat model, the conversation is a list of
messages, each with a *role*: `system`, `user`, or `assistant`. The **system** message is
special instructions that set the model's behavior and persona *before* the user says
anything — think of it as the model's job description. Ours tells the model to always call
the tool for real data and to **"Ground every claim in the tool's output — never invent
repository names, star counts, or descriptions."** That single sentence is our main defense
against the model *hallucinating* (making up) repositories. The prompt is passed into the
agent at [`agent.py:49`](../agent.py#L49).

---

## Concept 3: `build_llm` — constructing the chat model

[`agent.py:32-41`](../agent.py#L32-L41):

```python
def build_llm(settings: Settings) -> ChatNVIDIA:
    return ChatNVIDIA(
        model=settings.nvidia_model,
        api_key=settings.nvidia_api_key.get_secret_value(),   # unlock the secret here
        temperature=settings.llm_temperature,
        ...
    )
```

`ChatNVIDIA` is LangChain's client object for NVIDIA-hosted models — it's the thing that
actually sends your messages to the model and gets a reply. Notice
`.get_secret_value()`: as covered in [`tools.md`](tools.md), secrets are wrapped in
`SecretStr` so they can't leak into logs; we unwrap the real string **only** at the exact
moment we hand it to the client. This function is the single place the model key is unwrapped.

The `-> ChatNVIDIA` after the parentheses is the **return type hint**: "this function hands
back a `ChatNVIDIA`." (See Concept 5 for why the *next* function's return type is different.)

---

## Concept 4: What a "graph" is, and `create_react_agent`

[`build_agent`](../agent.py#L44) is the assembly line:

```python
def build_agent(settings: Settings) -> Any:
    llm = build_llm(settings)
    tools = build_github_tools(settings)
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
```

**"Graph" is just a fancy word for a wiring diagram.** Imagine boxes (called *nodes*) with
arrows between them (called *edges*). Each node does one step of work; the arrows say "after
this step, go to that step." A "**compiled** graph" is that diagram turned into a real,
runnable object you can call.

**`create_react_agent`** builds a specific, well-known graph shape called **ReAct**
(short for *Reason + Act*). It has two nodes and a loop:

```
   ┌──────────────┐   did the model ask    ┌───────────┐
   │  call model  │   to use a tool?        │ run tools │
   │  (ChatNVIDIA)│ ── yes ───────────────▶ │           │
   │              │ ◀───────────────────────│           │
   └──────────────┘   (feed tool result     └───────────┘
          │            back to the model)
          │ no — model wrote a plain answer
          ▼
        DONE
```

In words: send the messages to the model. If the model replies "please call
`list_github_repositories`," the graph runs that tool, appends the result to the
conversation, and loops back to the model. The model reads the tool's output and either calls
another tool or writes a final answer. When it writes a plain answer (no tool call), the loop
ends. We get all of this from **one line** — that's why we use the prebuilt helper instead of
hand-writing the loop.

---

## Concept 5: `Any` — the "turn off type-checking" type

Notice `build_agent` returns `Any` ([`agent.py:44`](../agent.py#L44)), not a specific type.

`Any` is a special type hint meaning *"this could be anything — type checker, please don't
complain about it."* It's an escape hatch. Normally you want precise types, but here the
object returned by `create_react_agent` has a complicated internal type from LangGraph that
changes between versions. Rather than fight it, we label it `Any` and re-establish safety at
the edges (see the `str(...)` cast in Concept 7).

Rule of thumb: `Any` is fine in small doses at the boundary with complex external libraries;
avoid sprinkling it through your own logic, because it silently disables the safety net.

---

## Concept 6: `BasicAgent` — a facade, and dependency injection

[`class BasicAgent`](../agent.py#L52) is a **facade**: a small, clean front door that hides
something complicated behind it. Users of the class see two easy methods (`ask`, `stream`)
and never have to know about LangGraph, state dicts, or streaming modes.

Look at the constructor, [`agent.py:55-57`](../agent.py#L55-L57):

```python
def __init__(self, settings: Settings, *, graph: Any | None = None) -> None:
    self._settings = settings
    self._graph: Any = graph if graph is not None else build_agent(settings)
```

Several things to unpack:

- **`__init__`** is the constructor — it runs when you write `BasicAgent(settings)`. Its job
  is to set up the object's internal state.
- **`self`** is the object being built; `self._graph = ...` stores a value *on* that object
  so other methods can use it later.
- **The `_` prefix** on `self._settings` / `self._graph` is a convention meaning "internal —
  please don't touch this from outside the class." Python doesn't enforce privacy; the
  underscore is a polite "keep out" sign.
- **The keyword-only `graph` parameter** (after the `*`, same idea as in `tools.py`) with a
  default of `None` is a **dependency-injection seam** — a deliberate hole where you can pass
  in a substitute. In production you don't pass it, so the object builds the real graph. In
  tests, you pass a fake graph, so no LLM is ever called. This mirrors the `client_factory`
  trick in `tools.py`.
- **The conditional expression** `graph if graph is not None else build_agent(settings)` is
  Python's one-line if/else (often called a *ternary*). It reads left-to-right as: *"use
  `graph` if it was provided, otherwise build a real one."* The general form is
  `VALUE_IF_TRUE if CONDITION else VALUE_IF_FALSE`:

  ```python
  status = "adult" if age >= 18 else "minor"
  ```

---

## Concept 7: `ask` — one-shot answers

[`agent.py:59-62`](../agent.py#L59-L62):

```python
def ask(self, question: str) -> str:
    result = self._graph.invoke(self._initial_state(question))
    return str(result["messages"][-1].content)
```

- **`.invoke(...)`** runs the whole ReAct loop start-to-finish and returns the final result
  *all at once* (as opposed to streaming it piece by piece — that's Concept 8).
- **What comes back** is a dictionary with a `"messages"` key holding the *entire*
  conversation: the system prompt, the user question, any tool calls and their results, and
  finally the model's answer.
- **`result["messages"][-1]`** grabs the **last** message in that list. The `[-1]` is
  **negative indexing** — Python lets you count from the end, so `[-1]` is the last item,
  `[-2]` the second-to-last, etc.:

  ```python
  [10, 20, 30][-1]   # → 30
  ```

  The last message is the model's final answer, and `.content` is its text.
- **`str(...)`** wraps the result to *guarantee* we return a real string. Remember from
  Concept 5 that the graph is typed `Any`, so `.content` is loosely typed; `str(...)`
  re-establishes a precise `-> str` return type at the boundary.

---

## Concept 8: `stream` — generators and `yield` (the big new idea)

[`agent.py:64-73`](../agent.py#L64-L73):

```python
def stream(self, question: str) -> Iterator[str]:
    for chunk, _metadata in self._graph.stream(
        self._initial_state(question), stream_mode="messages"
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        content = chunk.content
        if isinstance(content, str) and content:
            yield content
```

**What is a generator?** A normal function computes everything and `return`s one result. A
function that uses **`yield`** instead is a **generator**: it produces a *sequence* of values
lazily — one at a time, pausing after each one until you ask for the next.

```python
def count_up(n):
    i = 1
    while i <= n:
        yield i        # hand back ONE value, then pause right here
        i += 1

for x in count_up(3):
    print(x)           # prints 1, then 2, then 3 — produced one at a time
```

The key difference from building a list: a generator doesn't wait to have *all* the values
before giving you the first one. That's exactly what we want for streaming — as each piece
("token") of the model's answer arrives, we `yield` it immediately so the CLI can print it
live, letter by letter, instead of waiting for the whole answer.

**`Iterator[str]`** is the return type hint for "a stream of strings" — i.e. "this is a
generator that yields `str` values." (`Iterator` is imported at
[`agent.py:10`](../agent.py#L10).)

Now the rest of the loop, line by line:

- **`self._graph.stream(..., stream_mode="messages")`** asks the graph to emit its progress
  incrementally instead of all at once. `stream_mode="messages"` specifically means "give me
  the message pieces as the model generates them."
- **`for chunk, _metadata in ...`** — the stream yields **pairs** (Python *tuples*), and this
  line unpacks each pair into two variables at once. Tuple unpacking looks like:

  ```python
  for name, age in [("Sam", 30), ("Jo", 25)]:
      ...
  ```

  Here each item is `(message_piece, some_metadata)`. We want the first and ignore the
  second — so we name the second **`_metadata`**. A leading underscore is the Python
  convention for *"I'm required to receive this, but I'm intentionally not using it."*
- **`isinstance(chunk, AIMessageChunk)`** is a runtime type check: "is `chunk` an
  `AIMessageChunk`?" `isinstance(x, T)` returns `True`/`False`:

  ```python
  isinstance(3, int)      # True
  isinstance("hi", int)   # False
  ```

  Why do we check? Because the stream contains **every** kind of message piece — including
  the tool's output (a different type). We only want the *model's own words*, which are
  `AIMessageChunk`s. If a chunk isn't one, **`continue`** skips to the next loop iteration.
- **`if isinstance(content, str) and content:`** — a chunk's content is *usually* a string
  but can occasionally be other shapes, so we confirm it's a string; and `and content`
  skips empty pieces (an empty string is "falsy", so `and content` is `False` for `""`).
- Only when all checks pass do we **`yield content`** — handing that piece of the answer to
  whoever is looping over `stream(...)` (the CLI).

The test [`test_stream_yields_only_assistant_tokens`](../../../tests/test_agent.py)
deliberately injects a tool message into a fake stream to prove this filtering works — the
tool noise is dropped, only the model's words come out.

---

## Concept 9: `@staticmethod` and the message "state"

[`agent.py:75-77`](../agent.py#L75-L77):

```python
@staticmethod
def _initial_state(question: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": question}]}
```

**`@staticmethod`** is a *decorator* (a label starting with `@` placed above a function that
modifies it). It marks a method that lives inside the class for organization but **doesn't
use `self`** — it needs nothing from a particular instance. Notice there's no `self`
parameter. We made it static because turning a question into the starting state is pure
input→output logic that happens to belong, tidiness-wise, with this class.

**The return value** is the **starting state** the graph runs on:

```python
{"messages": [{"role": "user", "content": question}]}
```

Recall from Concept 2 that a conversation is a list of role-tagged messages. Here we seed the
conversation with a single **`user`** message containing the question. (The `system` prompt is
already baked into the graph, so we don't repeat it.) The ReAct loop then *grows* this
`messages` list — adding the model's tool calls, the tool results, and the final answer — as
it runs. `dict[str, Any]` as the return type means "a dictionary with string keys and values
of any type."

---

## The whole file in one paragraph

`build_llm` creates the `ChatNVIDIA` chat model (unwrapping the API key only at that
boundary). `build_agent` hands the model, the GitHub tool, and a **system prompt** to
`create_react_agent`, which compiles a **ReAct graph** — a two-node loop that alternates
between "ask the model" and "run the tool it requested" until the model produces a plain
answer. Because that compiled graph has a messy library type, we label it `Any` and restore
precise types at the edges. `BasicAgent` is a **facade** exposing just `ask` (run the loop
and return the final message via `.invoke`) and `stream` (a **generator** that `yield`s the
model's answer token-by-token via `.stream`, filtering out non-model pieces). The optional
`graph` constructor parameter is a **dependency-injection seam** that lets tests swap in a
fake graph so no real LLM is ever called.

---

See also: [`tools.md`](tools.md) — how the tool this agent calls is built.
