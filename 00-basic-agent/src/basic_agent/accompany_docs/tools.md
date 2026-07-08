# Companion notes: `tools.py`

> A junior-friendly, concept-by-concept walkthrough of [`../tools.py`](../tools.py).
> It explains the Python and Pydantic ideas used in that file — `BaseModel`, `Protocol`,
> context managers, closures, type aliases — with tiny throwaway examples, then points
> back at the real code.

`tools.py` is small, but it packs in several concepts that aren't obvious until someone
explains them. We go concept-by-concept from the ground up.

---

## Concept 1: `BaseModel` and `Field` (from Pydantic)

Start with the most familiar-looking part: [`ListRepositoriesInput`](../tools.py#L50).

Normally in Python, a small data-holding class means a lot of boilerplate:

```python
class ListRepositoriesInput:
    def __init__(self, username=None, include_forks=False, limit=20):
        # you'd have to hand-write all the validation:
        if not isinstance(limit, int):
            raise TypeError("limit must be an int")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be 1..100")
        self.username = username
        self.include_forks = include_forks
        self.limit = limit
```

Pydantic's `BaseModel` does all of that **for you, automatically, from the type hints**.
You just declare what the fields are:

```python
class ListRepositoriesInput(BaseModel):
    username: str | None = None
    include_forks: bool = False
    limit: int = 20
```

Now Pydantic auto-generates the `__init__`, and **validates and converts** input:

```python
ListRepositoriesInput(limit="5")     # → limit becomes the integer 5 (string parsed)
ListRepositoriesInput(limit="abc")   # → raises ValidationError, clean message
ListRepositoriesInput()              # → uses the defaults
```

So a `BaseModel` is **"a class that describes a shape of data and enforces it."**

**What is `Field(...)`?** A field's type gives it a type; `Field` lets you attach *extra*
information to it. Look at [`tools.py:61-66`](../tools.py#L61-L66):

```python
limit: int = Field(
    default=20,        # the value if none is provided
    ge=1,              # "greater-or-equal 1"
    le=100,            # "less-or-equal 100"
    description="Maximum number of repositories to return.",
)
```

- `default=20` — same as writing `= 20`, just the explicit form.
- `ge` / `le` — validation rules. Pydantic now *enforces* `1 ≤ limit ≤ 100`.
- `description` — human-readable text. In *this* file that description is special: it's
  shipped to the LLM so the model knows what the argument means (see Concept 7).

So `username: str | None = Field(default=None, description=...)` at
[`tools.py:53`](../tools.py#L53) just means: *an optional string, defaulting to nothing,
with a description attached.*

Quick note on the type `str | None`: the `|` means "or". `str | None` = "either a string
or `None`" — Python's way of saying "this is optional / can be empty."

---

## Concept 2: `Protocol` — a shape, not a family tree

This is [`RepositoryLister`](../tools.py#L21), and it's the idea most worth slowing down on.

**The problem it solves.** The tool needs *something* that can fetch repos. The obvious way
is to require our real `GitHubClient`:

```python
def list_github_repositories(...):
    with GitHubClient(...) as client:   # hard-wired to the real thing
        ...
```

But then how do you test it without hitting the real GitHub over the network? You can't
easily swap in a fake.

**Two ways to say "something that behaves like X":**

The traditional (Java/C#-style) way is **inheritance** — a "family tree." You'd make a base
class and force every client to inherit from it:

```python
class BaseClient(ABC): ...
class GitHubClient(BaseClient): ...   # must explicitly inherit
class FakeClient(BaseClient): ...     # must explicitly inherit
```

The Python-idiomatic way is a **`Protocol`** — a "shape" or "checklist." It says: *"I don't
care what class you are or what you inherit from. If you have these methods with these
signatures, you qualify."*

This is **duck typing** ("if it walks like a duck and quacks like a duck, it's a duck") —
but with a type checker that can verify it *before* you run the program.

A minimal example:

```python
from typing import Protocol

class Greeter(Protocol):
    def greet(self, name: str) -> str: ...   # just the shape, no body

class French:                     # note: does NOT inherit from Greeter
    def greet(self, name): return f"Bonjour {name}"

class Dog:
    def bark(self): return "woof"

def welcome(g: Greeter) -> str:   # "give me anything shaped like a Greeter"
    return g.greet("Sam")

welcome(French())   # ✅ has greet(name) → matches the shape
welcome(Dog())      # ❌ mypy error: Dog has no greet method
```

`French` works even though it never mentions `Greeter`. That's the magic: the match is
**structural** (based on shape), not **nominal** (based on names in a family tree).

Now re-read [`RepositoryLister`](../tools.py#L21). It says: *"To be usable by this tool, you
must support `with` (the `__enter__`/`__exit__` methods) and have a
`list_public_repositories(...)` method."* Our real
[`GitHubClient`](../github_client.py#L48) has exactly those methods, so it qualifies —
**without importing or inheriting `RepositoryLister` at all**. And the `FakeClient` in the
tests also has those methods, so *it* qualifies too. That's precisely why the tests can run
with no network.

**The `...` in the Protocol methods** (e.g. [`tools.py:29`](../tools.py#L29)): `...` is
literally a Python object called `Ellipsis`, used here to mean "no body — this is just the
signature." It's like `pass`. A Protocol only describes shapes; it never contains real
logic.

---

## Concept 3: Context managers — the machinery behind `with`

The Protocol mentions `__enter__` and `__exit__`. These are the two "magic methods" that
make an object usable in a `with` block. You've almost certainly *used* `with` before:

```python
with open("file.txt") as f:
    data = f.read()
# file is automatically closed here, even if read() crashed
```

What `with X as y:` actually does under the hood:

1. calls `X.__enter__()` and assigns its return value to `y`
2. runs the indented block
3. **guarantees** it calls `X.__exit__(...)` afterward — even if an exception was raised

`__exit__` is where cleanup lives (closing files, network connections, etc.). That's why
our client uses it: [`tools.py:124`](../tools.py#L124) `with factory() as client:` opens an
HTTP connection pool and is *guaranteed* to close it when the block ends, even if the
request throws. You never leak connections.

The scary-looking `__exit__` signature at [`tools.py:31-36`](../tools.py#L31-L36):

```python
def __exit__(
    self,
    exc_type: type[BaseException] | None,   # the *class* of the error, if one happened
    exc: BaseException | None,              # the actual error *object*, if one happened
    tb: TracebackType | None,               # the traceback, if one happened
) -> None: ...
```

Python passes these three arguments automatically. If the `with` block finished cleanly, all
three are `None`. If it blew up, they describe the error so `__exit__` can decide what to do.
You don't call this yourself — Python does.

One subtlety: `type[BaseException]` vs `BaseException`. `BaseException` = an *instance* (an
actual error that happened). `type[BaseException]` = the *class itself* (like `ValueError`
the class, not a specific `ValueError("oops")`). And `| None` on each just means "or
nothing, if there was no error."

**`Self`** at [`tools.py:29`](../tools.py#L29): `__enter__(self) -> Self` means "returns an
instance of this same class." It's a precise way to say "`with GitHubClient() as c` gives you
back a `GitHubClient`."

---

## Concept 4: `Callable` and type aliases

Line [`tools.py:47`](../tools.py#L47):

```python
ClientFactory = Callable[[], RepositoryLister]
```

Two things here.

**`Callable[[args], return]`** is the type for "a function." It reads:
`Callable[[list of argument types], return type]`.

- `Callable[[], RepositoryLister]` = "a function that takes **no** arguments (`[]` is empty)
  and **returns** a `RepositoryLister`."
- `Callable[[int, str], bool]` would be "a function taking an int and a str, returning a
  bool."

**A type alias.** `ClientFactory = ...` just gives that long type a short, meaningful name.
Instead of writing `Callable[[], RepositoryLister]` everywhere, we write `ClientFactory`.
It's purely for readability — like naming a variable. So "a `ClientFactory`" now means "a
no-argument function that hands you back something that can list repos."

Why pass a *function that makes a client* rather than a client directly? Because a client
holds a live connection. We only want to create it at the moment we actually use it (and
close it right after). A factory lets the caller say *how* to build one, while the tool
controls *when*. That pays off in Concept 6.

---

## Concept 5: `from __future__ import annotations` (line 8)

Small but ubiquitous. [`tools.py:8`](../tools.py#L8). This line tells Python: *"treat all my
type hints as text, don't try to evaluate them at runtime."*

Two practical benefits:

1. It's slightly faster and lets you reference types defined later in the file.
2. It's what lets us write modern syntax like `str | None` and `list[GitHubRepository]`
   freely without older-Python evaluation quirks.

You can treat it as a "turn on the nice modern type-hint behavior" switch that goes at the
top of every module. Type hints are for humans and tools (mypy, your editor) — they don't
change how the code runs.

---

## Concept 6: The factory function and **closures**

Now the heart of the file: [`build_github_tools`](../tools.py#L89). This is a function that
*builds and returns* tools. Two ideas make it work.

**Keyword-only arguments (the `*`).** Look at the signature
[`tools.py:89-93`](../tools.py#L89-L93):

```python
def build_github_tools(
    settings: Settings,
    *,
    client_factory: ClientFactory | None = None,
) -> list[BaseTool]:
```

The lone `*` means "everything after me must be passed by name, not by position." So you
must call `build_github_tools(settings, client_factory=something)` — you *cannot* write
`build_github_tools(settings, something)`. This prevents mix-ups and makes call sites
self-documenting. (You'll see the same `*` in the client's `list_public_repositories`.)

**Closures.** This is the big one. A **closure** is an inner function that "remembers"
variables from the outer function it was defined in.

```python
def make_multiplier(n):
    def multiply(x):
        return x * n      # 'n' is remembered from the outer function
    return multiply

double = make_multiplier(2)
double(10)   # → 20   ('double' still remembers n=2)
```

`build_github_tools` uses this twice:

- `_default_factory` at [`tools.py:102`](../tools.py#L102) is an inner function that
  **remembers `settings`** — it reads `settings.github_token`, `settings.github_api_base_url`,
  etc. It's how "build a real client" gets access to config without config being a global.
- `list_github_repositories` at [`tools.py:112`](../tools.py#L112) also remembers `settings`
  (for the default username at [`tools.py:117`](../tools.py#L117)) **and** `factory` (to make
  a client at [`tools.py:124`](../tools.py#L124)).

Why do it this way? Because the LLM will call
`list_github_repositories(username=..., limit=...)`. We do **not** want the LLM to supply the
API token or base URL — those are our concern, not the model's. The closure lets the inner
function quietly access config while exposing only the clean `username / include_forks /
limit` interface to the outside world.

**The `or` trick** at [`tools.py:110`](../tools.py#L110):

```python
factory = client_factory or _default_factory
```

In Python, `A or B` returns `A` if `A` is "truthy", otherwise `B`. `None` is "falsy". So this
reads: *"use the injected `client_factory` if the caller gave one; otherwise fall back to
`_default_factory`."* In production nobody passes one → we build a real client. In tests, the
suite passes `client_factory=lambda: FakeClient(...)` → we build a fake. Same line, both
behaviors. (A `lambda` is just a one-line anonymous function; `lambda: FakeClient()` is "a
no-arg function that returns a FakeClient" — exactly a `ClientFactory`.)

**Putting the inner function together** ([`tools.py:112-130`](../tools.py#L112-L130)):

```python
resolved = (username or settings.github_username or "").strip()
```

Same `or` chaining: use the LLM-provided username, else the configured default, else `""`
(empty). `.strip()` removes stray whitespace. If after all that it's still empty
([`tools.py:118`](../tools.py#L118)), we return a polite message instead of crashing.

Then:

```python
try:
    with factory() as client:                       # make a client, guaranteed-close
        repos = client.list_public_repositories(...) # fetch
except GitHubError as exc:                            # any GitHub failure...
    return f"Could not fetch repositories for {resolved}: {exc}"  # ...becomes text
return _format_repositories(resolved, repos)
```

Notice: **the tool never raises**. Success → formatted text. Failure → an explanatory
string. That's intentional — the tool's output is fed back to the LLM, and a sentence like
"Could not fetch repositories: user not found" is something the model can read and react to,
whereas an uncaught exception would crash the whole agent.

---

## Concept 7: Turning a plain function into a "tool" the LLM can call

Last piece: [`tools.py:132-142`](../tools.py#L132-L142).

```python
tool = StructuredTool.from_function(
    func=list_github_repositories,   # the Python function to run
    name="list_github_repositories", # the name the LLM uses to call it
    description="List a GitHub user's public repositories...",  # when to use it
    args_schema=ListRepositoriesInput,  # the arguments' shape (our BaseModel!)
)
return [tool]
```

An LLM can't run Python. It can only *emit text* like "I'd like to call
`list_github_repositories` with `limit=5`." Something has to (a) tell the model what tools
exist and what their arguments are, and (b) actually run the real function when the model
asks. `StructuredTool.from_function` builds that bridge:

- `name` + `description` are what the model reads to decide *whether and when* to use this
  tool.
- `args_schema=ListRepositoriesInput` — this is where the `BaseModel` from Concept 1 comes
  back. LangChain turns that model (fields, types, and their `description`s) into a JSON
  schema handed to the LLM, so the model knows the tool takes an optional `username`, a
  boolean `include_forks`, and an integer `limit` between 1 and 100. When the model sends
  arguments back, Pydantic validates them against that same model before your function runs.

We return `[tool]` — a **list** — because the return type is
[`list[BaseTool]`](../tools.py#L93). `list[BaseTool]` means "a list whose items are
`BaseTool` objects" (`BaseTool` is LangChain's base type for all tools). Today there's one
tool; returning a list means later agents can add more without changing this function's shape.

---

## The whole file in one paragraph

`build_github_tools(settings)` is a factory that produces the LLM-callable tool(s). It uses a
**closure** so the tool can quietly access config (`settings`) and a client-builder
(`factory`) while exposing only a clean `username/include_forks/limit` interface. That
interface's shape is described by a Pydantic **`BaseModel`** (`ListRepositoriesInput`), which
doubles as the schema the LLM reads. The tool depends on a **`Protocol`**
(`RepositoryLister`) — a structural "shape" — instead of the concrete client, so real code
and test fakes are interchangeable. Client lifetime is handled by a **context manager**
(`with`), guaranteeing connections close. And every failure is converted to a string so the
LLM can cope, never a crash.

---

See also: [`agent.md`](agent.md) — how these tools get wired to the LLM in a ReAct loop.
