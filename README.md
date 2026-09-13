# RegDocs Q&A

Ask plain-English questions about dense financial regulations and get back an
answer that cites the document and page it came from. Regulatory PDFs are long,
cross-referenced, and written for specialists: the three documents in this
project run to over 1,100 pages, and finding the rule you need usually means
knowing where to look before you start. This tool removes that requirement. You
ask a question in your own words, and it finds the relevant passages by meaning
rather than by keyword, then has Claude write the answer using only those
passages. Every claim carries a filename and page number, so you can go and
check the source yourself. When the documents do not contain the answer, it says
so instead of guessing, which matters more here than anywhere: a confident wrong
answer about a regulation is worse than no answer at all.

## Architecture

```
docs/*.pdf                 three regulatory PDFs, 1,169 pages of text
    |
    |  ingest.py           extract text page by page with pypdf,
    v                      split into ~400-word chunks, 50-word overlap
chunks.json                2,162 chunks, each tagged with source file + page
    |
    |  embed.py            Voyage voyage-4, input_type="document"
    v                      each vector scaled to length 1
embeddings.npy             2,162 x 1,024 numbers
    |
    |  search.py           embed the question (input_type="query"),
    v                      score all chunks with one numpy dot product
top 3 chunks               ranked by cosine similarity
    |
    |  ask.py              pass chunks to Claude with a grounding prompt
    v
answer + citations
```

Three details are worth knowing:

**Chunking happens per page, not per document.** A short page becomes a short
chunk, which is slightly wasteful, but it means every chunk keeps an accurate
page number and the final answer can cite a real page.

**Vectors are normalised at embed time.** Cosine similarity is
`(a · b) / (|a| × |b|)`. Scaling every vector to length 1 in advance makes both
divisors 1, so searching reduces to a plain dot product. All 2,162 comparisons
happen in a single numpy operation.

**Row N of `embeddings.npy` matches chunk N of `chunks.json`.** Nothing enforces
that but the order the files were written in, so rebuild them as a pair.
`search.py` checks the counts match and refuses to run if they do not.

## Setup

Requires a conda environment named `regdocs` with `pypdf`, `numpy`, `voyageai`,
`anthropic`, and `python-dotenv` installed.

1. Put your PDFs in `docs/`.
2. Create a `.env` file in the project root:

   ```
   ANTHROPIC_API_KEY=your-key-here
   VOYAGE_API_KEY=your-key-here
   ```

   `.env` is listed in `.gitignore` and must stay there.

3. Build the chunks and the embeddings:

   ```
   conda run -n regdocs python ingest.py
   conda run -n regdocs python embed.py
   ```

## Usage

Ask a question:

```
conda run -n regdocs python ask.py "how long does a consumer have to report an error?"
```

To see the raw matching passages without an answer written over them:

```
conda run -n regdocs python search.py "how long does a consumer have to report an error?"
```

### A note on rate limits

A Voyage account with no payment method allows 3 requests and 10,000 tokens per
minute. Every question costs one request, so asking several in quick succession
will pause and retry. Building `embeddings.npy` from scratch takes about two
hours under that limit; `embed.py` paces itself, saves progress after every
request, and resumes if interrupted. Adding a payment method and setting
`FREE_TIER = False` in `embed.py` reduces the full run to a couple of minutes.

## Examples

Real output, lightly trimmed for length.

**Q: How many days does a consumer have to report an error, and what must the
financial institution do after receiving notice?**

> A consumer must report an error not later than **60 days** after the financial
> institution sends the periodic statement or other documentation first
> reflecting the alleged error (12 CFR 1005.11(b)(1)) (consumer compliance
> handbook.pdf, p. 97).
>
> After receiving a notice of error, the financial institution must promptly
> investigate the allegation, complete its investigation within 10 business
> days, report the results within three business days of completing it, and
> correct the error within one business day of determining one occurred
> (consumer compliance handbook.pdf, p. 97).
>
> The institution may take up to **45 calendar days** instead, provided it
> provisionally credits the funds within the 10 business-day period, advises the
> consumer within two business days, and gives full use of the funds during the
> investigation (consumer compliance handbook.pdf, p. 97).

**Q: What risks should a financial institution manage in retail payment
systems?**

> Financial institutions should manage the following risks in retail payment
> systems: strategic, credit, liquidity, settlement, legal/compliance, and
> operational/transaction risk (FFIEC retail payment systems booklet pdf.pdf,
> p. 5).
>
> Risk management processes should "capture not only operational risks, but also
> credit, liquidity, strategic, legal, and compliance risks, particularly as
> they engage in new retail payment products and systems" (FFIEC retail payment
> systems booklet pdf.pdf, p. 38).

**Q: What is the recommended oil change interval for a 2019 Honda Civic?**

> Not found in the documents

That last one is the most important example in this README. The question has
nothing to do with the documents, and the system says so rather than inventing
an answer. The retrieved passages scored around 0.33 on similarity, against 0.6
to 0.7 for the genuine questions, so the low scores are themselves a signal that
nothing relevant was found.

## Evaluation

A 15-question eval lives in `evals/`. Run it with:

```
conda run -n regdocs python evals/run.py
```

**Methodology.** Each question carries a reference answer and the document that
should contain it, and the suite measures three things: whether that document
appears in the top 3 retrieved chunks, whether Claude's answer conveys the same
key facts as the reference (graded by Claude acting as a judge), and whether the
three deliberately out-of-scope questions are refused while the other twelve are
not. The questions span easy lookups, harder multi-part questions, and
unanswerable ones, and the reference answers were drafted by running the real
pipeline and then editing for correctness against the source documents.

**Scores.**

| Measure | Score | |
|---|---|---|
| Retrieval hit rate | 12/12 | 100% |
| Answer correctness | 11/12 | 92% |
| Refusal accuracy | 15/15 | 100% |
| Overall | 14/15 | 93% |

**The one failure.** Question 6 asks when an institution can extend an error
investigation to 45 days and what it must do first. The answer correctly
identified provisional credit as the prerequisite but omitted two further
conditions: informing the consumer within two business days, and giving full use
of the funds during the investigation. Those conditions exist in the corpus and
were retrieved for question 2, but did not surface in question 6's top 3 chunks.
Notably, Claude hedged rather than inventing them, writing that the requirement
was "implied by" the passages. The grounding held; the retrieval was too
shallow.

**A near miss worth recording.** Question 7 passed, but for a shakier reason
than the score suggests. Its top two chunks were pages 584 and 585 of the
consumer compliance handbook, which cover mortgage servicing errors under RESPA
and Regulation X, not electronic transfer errors under Regulation E. The correct
Regulation E material appeared only in the third slot. The retrieval metric
still recorded a hit, because it checks the filename and both regulations live
in the same PDF. A document-level metric is too coarse to catch this.

### Known limitations

1. **Regulations are not distinguished within a document.** Question 7 retrieved
   RESPA and Regulation X passages for a Regulation E question, because error
   resolution language is worded almost identically across the two regimes. The
   chunks carry a filename and page but no indication of which regulation they
   belong to, so neither retrieval nor the eval metric can tell them apart.

2. **Retrieval depth and chunk size limit multi-part answers.** Question 6
   needed several conditions that are spread across a page, and three chunks of
   roughly 400 words did not capture all of them. Questions whose answers are
   enumerated lists are the weak spot, and returning more chunks or sizing them
   differently would likely help.

3. **Similarity alone cannot gate answerability.** Out-of-scope questions scored
   0.377 to 0.428, while the weakest genuine questions scored 0.514 and 0.515.
   The gap is real but only 0.086 wide, so any cutoff placed between them would
   be fragile. Correct refusals here come from the grounding prompt, not from
   the score, and the score should not be trusted to do that job.

One further caveat about the method itself: the judge is the same model family
that writes the answers, which tends to grade generously. Treat 92% as a signal
to track across changes rather than an absolute measure of quality.

## Next steps

- **Metadata filtering by regulation** — tag each chunk with the regulation it
  belongs to, so a Regulation E question cannot retrieve Regulation X passages.
  This directly addresses limitation 1.
- **Reranking** — pass a wider set of candidates, say the top 20, through a
  reranking model that scores relevance more precisely than cosine similarity,
  then keep the best 3. This would have promoted the Regulation E material in
  question 7 above the mortgage servicing text.
- **Chunk-size tuning** — evaluate different chunk sizes, overlaps, and values
  of top-N against the eval suite, targeting the enumerated-list answers that
  question 6 exposed as the weak spot.
- **Chroma** — replace the hand-rolled numpy search with a real vector database,
  so lookups stay fast as the document set grows beyond a few thousand chunks.
- **Streamlit** — a simple web interface, so someone can ask questions without
  touching a terminal.
- **Docker** — package the whole thing so it runs identically on any machine.

## Files

| File | What it is |
|---|---|
| `ingest.py` | PDFs to `chunks.json` |
| `embed.py` | `chunks.json` to `embeddings.npy` via Voyage |
| `search.py` | Finds the chunks closest in meaning to a question |
| `ask.py` | Answers a question with Claude, citing sources |
| `chunks.json` | 2,162 chunks with source file and page |
| `embeddings.npy` | 2,162 vectors of 1,024 numbers |
| `evals/questions.json` | 15 test questions with reference answers |
| `evals/run.py` | Runs the eval and prints the scorecard |
| `evals/results.json` | Latest eval output |
| `docs/` | Input PDFs (not tracked in git) |
| `.env` | API keys (never committed) |
