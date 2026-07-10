# 02 · Chunking — why we split documents

## The problem with embedding a whole document

An embedding squeezes a piece of text into **one** vector. That works for a sentence or a
paragraph. But a whole README covers many topics — architecture, setup, security, CI/CD. Cram
all of that into a single vector and you get a blurry average that's close to *nothing* in
particular. Worse, when you retrieve it, you hand the model the entire document when it only
needed one paragraph.

## The fix: chunks

So before embedding, we **split each document into small passages ("chunks")** — roughly a
few paragraphs each. Each chunk gets its own vector. Now retrieval can return *just* the
paragraph about medallion layers, not the whole 400-line README.

Two settings control this:

- **chunk size** — how big each passage is.
- **chunk overlap** — how much consecutive chunks share, so an idea that straddles a boundary
  isn't sliced in half and lost.

## The trade-off you're balancing

- **Chunks too big** → back to the blurry-average problem; retrieval is imprecise and you
  waste prompt space.
- **Chunks too small** → each piece loses the context around it; a chunk saying "it uses
  cosine distance" is useless if you can't tell *what* "it" is.

There's no universal right answer — it depends on your documents, which is why chunk size and
overlap are **config knobs we can tune later** and measure (see [08](08-evaluating-retrieval.md)).

## Chunks carry metadata

When we split a document we tag every chunk with where it came from — its **source file** and
**repo**. That tag is small but crucial: it's what lets us later say "only search within the
carthage repo" ([06](06-metadata-filtering.md)) and attribute an answer to a specific project.

## A real lesson from this project

When we debugged a failure, retrieval once returned a chunk that was just a slab of
**ASCII-art diagram** from one README. That's a chunking symptom: naive splitting can produce
low-value "noise" chunks. Better, structure-aware chunking (splitting on headings, dropping
noise) is a real improvement lever — the kind of thing an eval set tells you is worth doing.

## Where this lives

`../ingest.py` (`split_documents`) does the splitting; `../config.py` holds the knobs.
