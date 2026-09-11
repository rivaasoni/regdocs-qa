# RegDocs Q&A

A small Retrieval-Augmented Generation (RAG) project that answers questions
about regulatory PDF documents and cites the page it got each answer from.

This file is the project plan and the working agreement. Read it first.

## How to work on this project

These rules matter more than speed. The author is new to coding.

- **Keep the code simple.** Plain Python, small functions, obvious names.
- **Comment generously.** Explain *why* a line exists, not just what it does.
- **No frameworks.** Do not use LangChain, LlamaIndex, or similar. Call the
  Anthropic and Voyage APIs directly so every step stays visible.
- **One phase at a time.** Do not start the next phase until asked.

## Environment

- **Python environment:** conda env named `regdocs`.
  Run everything with `conda run -n regdocs python <script>.py`.
- **Secrets:** `.env` in the project root holds `ANTHROPIC_API_KEY` and
  `VOYAGE_API_KEY`. `.env` is listed in `.gitignore` and must stay there.
  Never print, log, or commit key values.
- **Input documents:** PDFs go in `docs/`. They are the source of truth for
  every answer the app gives.
- **Models:** Claude `claude-sonnet-5` for answering. Voyage for embeddings;
  confirm the current embedding model name at docs.voyageai.com before use.

## The five phases

### Phase 1 - Environment setup (done)

Confirm `.env` exists and holds both API keys, confirm `docs/` holds the PDFs,
and confirm `.env` is ignored by git so the keys never reach GitHub.

### Phase 2 - Ingestion and chunking (code done)

`ingest.py` reads every PDF in `docs/`, extracts the text page by page with
pypdf, and splits it into chunks of about 400 words with 50 words of overlap.
The overlap means an idea sitting on the seam between two chunks still appears
whole in at least one of them. Each chunk is a dictionary:

```python
{"text": "...", "source": "some-rule.pdf", "page": 7}
```

Chunking happens per page rather than per document, so every chunk keeps an
accurate page number and later phases can cite a real page. All chunks are
saved to `chunks.json`, and the script prints the total count plus one sample.

### Phase 3 - Embedding and retrieval

Turn each chunk's text into an embedding (a list of numbers capturing its
meaning) using the Voyage API. Store the vectors alongside the chunks. To
answer a question, embed the question the same way and compare it to every
chunk vector using cosine similarity computed with numpy. Cosine similarity
measures the angle between two vectors, so it scores chunks by meaning rather
than by shared keywords. Return the closest chunks.

### Phase 4 - Asking questions

`ask.py` takes a question, retrieves the top 3 chunks from Phase 3, and passes
them to Claude as context with an instruction to answer only from those chunks
and to cite the source file and page. If the chunks do not contain the answer,
Claude should say so rather than guess, since a confident wrong answer about a
regulation is worse than no answer. Also write a `README.md` explaining what
the project does and how to run it.

### Phase 5 - Making it real

Four upgrades, in whatever order makes sense:

- **Chroma** - replace the hand-rolled numpy search with a real vector
  database, so lookups stay fast as the document set grows.
- **Evals** - a set of questions with known-correct answers, run automatically,
  to measure whether changes make the system better or worse.
- **Streamlit** - a simple web interface so someone can ask questions without
  touching a terminal.
- **Docker** - package the app so it runs the same way on any machine.

## Files

| File | What it is |
|---|---|
| `ingest.py` | Phase 2. PDFs to `chunks.json`. |
| `chunks.json` | Generated output. Rebuild it by rerunning `ingest.py`. |
| `docs/` | Input PDFs. |
| `.env` | API keys. Never committed. |
