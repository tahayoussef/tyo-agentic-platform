# 00 · What RAG is (and the problem it solves)

## The problem

A language model knows two kinds of things: general knowledge baked into it during training,
frozen at some cutoff date. It does **not** know *your* private data (your repos, your docs),
and when asked about something it hasn't seen, it will often **make up** a confident,
plausible, wrong answer (a "hallucination").

## The idea

**RAG = Retrieval-Augmented Generation.** Instead of hoping the model already knows, you:

1. **Retrieve** the handful of most relevant snippets from your own data, then
2. paste them into the prompt, and
3. ask the model to answer **using those snippets**.

The model stops guessing and starts summarizing real text you handed it. That's the whole
trick — the "retrieval" step is what the rest of these docs are about.

## Our twist: two sources of truth

This agent answers questions about a set of GitHub repositories using **two** sources:

- a **static knowledge base** — curated repo documentation we indexed ahead of time (rich
  background, but can go stale), and
- the **live GitHub API** — current facts like stars and primary language (fresh, but shallow).

The agent pulls from both and **reconciles** them, flagging where the static docs disagree
with reality. The static half is a classic vector-database RAG pipeline; that pipeline is
what you'll learn to build and use across these files.

## Why bother with a whole database for this?

Because "find the most relevant snippets" is hard when you have thousands of them. You can't
keyword-match (synonyms and phrasing get in the way), and you can't stuff every document into
the prompt (too big, too expensive). You need a way to search by **meaning**, fast. That
starts with turning text into vectors → next file.
