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
- **Models:** Claude `claude-sonnet-5` for answering. Voyage `voyage-4` for
  embeddings, which returns 1024 numbers per chunk.
- **Voyage rate limits:** an account with no payment method is capped at 3
  requests and 10,000 tokens per minute. Our corpus is ~927,000 tokens, so a
  full embed run takes ~100 minutes on the free tier. `embed.py` paces itself
  and saves progress so it can resume. Adding a payment method and setting
  `FREE_TIER = False` in `embed.py` cuts the run to a couple of minutes.

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

### Phase 3 - Embedding and retrieval (code done)

`embed.py` turns each chunk's text into an embedding, a list of 1024 numbers
capturing its meaning, using Voyage's `voyage-4` model with
`input_type="document"`. Every vector is scaled to length 1 before saving, so
cosine similarity later reduces to a plain dot product. The result is saved to
`embeddings.npy`, where row N lines up with chunk N of `chunks.json`. That
ordering is the only thing tying the two files together, so they must always
be rebuilt as a pair.

`search.py` embeds your question with `input_type="query"`, then scores it
against all 2,162 chunks in one numpy multiplication and returns the closest
three. Cosine similarity measures the angle between two vectors, so it ranks by
meaning rather than shared keywords. A question about "moving money
electronically" finds a passage about "electronic fund transfers".

`embed.py` will not re-embed an up-to-date `embeddings.npy`, because doing so
costs real money. Delete that file to force a rebuild.

### Phase 4 - Asking questions (done)

`ask.py` takes a question, reuses `search.py` to retrieve the top 3 chunks, and
passes them to Claude `claude-sonnet-5` as numbered context labelled with each
passage's filename and page. The system prompt requires Claude to answer only
from those passages, cite filename and page for every claim, and reply exactly
`Not found in the documents` when the passages do not contain the answer.
Verified: an off-topic question returns that refusal, with retrieval scores
around 0.33 against 0.6-0.7 for genuine questions.

Do not pass `temperature` to `claude-sonnet-5`; it is rejected with a 400.
Leaving `thinking` unset lets Claude decide how much to think, which is the
sensible default here.

`README.md` documents the project for people arriving at the repository.

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
| `embed.py` | Phase 3a. `chunks.json` to `embeddings.npy` via Voyage. |
| `search.py` | Phase 3b. Finds the chunks closest to a question. |
| `ask.py` | Phase 4. Answers a question with Claude, citing sources. |
| `README.md` | Project documentation for people reading the repo. |
| `embeddings.npy` | 2,162 vectors of 1,024 numbers. Tracked in git. |
| `docs/` | Input PDFs. |
| `.env` | API keys. Never committed. |
