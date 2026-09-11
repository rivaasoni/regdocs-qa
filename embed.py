"""
embed.py - Phase 3a of RegDocs Q&A

What this file does, in one sentence:
    It reads chunks.json, sends every chunk's text to the Voyage API to be
    turned into an "embedding", and saves all those embeddings to
    embeddings.npy so search.py can use them.

What on earth is an embedding?
    It is a list of numbers that captures the *meaning* of a piece of text.
    Voyage turns each chunk into 1024 numbers. Texts that mean similar things
    end up with similar lists of numbers, even when they share no words at all.
    That is what lets us search by meaning rather than by keyword, so a question
    about "moving money electronically" can still find a chunk that only ever
    says "electronic fund transfers".

About rate limits (important, please read):
    A Voyage account with no payment method is limited to 3 requests and 10,000
    tokens per minute. Our documents are about 927,000 tokens, so a full run on
    the free tier takes roughly 100 minutes. This script is built to cope with
    that. It keeps each request under the token limit, waits between requests so
    it never trips the limit, and saves its progress as it goes. If it stops for
    any reason, run it again and it picks up exactly where it left off.

    If you add a payment method at dashboard.voyageai.com, set FREE_TIER below
    to False and the same run finishes in a couple of minutes instead.

Run it with:
    conda run -n regdocs python embed.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

CHUNKS_FILE = Path("chunks.json")
OUTPUT_FILE = Path("embeddings.npy")

# Progress is saved here after every request so a long run can be interrupted
# and resumed. It is deleted automatically once the full run finishes.
PARTIAL_FILE = Path("embeddings_partial.npy")

MODEL = "voyage-4"  # Voyage's current general-purpose embedding model

# Set this to False once you have added a payment method to your Voyage
# account. It only changes how patiently we wait between requests.
FREE_TIER = True

if FREE_TIER:
    # These two numbers were found by experiment, not guesswork. On the free
    # tier a request of ~5,000 tokens succeeds and one of ~8,600 is rejected,
    # even though the advertised ceiling is 10,000 per minute. So we keep each
    # request well under that, and send under two per minute, which stays
    # inside both the 10,000-tokens and 3-requests limits with room to spare.
    TOKENS_PER_REQUEST = 4_000
    SECONDS_BETWEEN = 32
else:
    TOKENS_PER_REQUEST = 100_000
    SECONDS_BETWEEN = 0

MAX_TEXTS_PER_REQUEST = 128  # a hard limit of the Voyage API itself
MAX_RETRIES = 6  # rate limits are the usual failure, and they clear with time


def load_chunks():
    """
    Read chunks.json back into a list of dictionaries.

    This is the output of Phase 2. If it is missing, the fix is to run
    ingest.py, so we say that plainly instead of showing a stack trace.
    """
    if not CHUNKS_FILE.is_file():
        raise SystemExit(
            f"ERROR: {CHUNKS_FILE} not found.\n"
            "Run 'conda run -n regdocs python ingest.py' first."
        )
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def plan_batches(texts, token_counts):
    """
    Group the chunks into batches that each fit inside one API request.

    Two ceilings apply at once: a request may hold at most 128 texts, and on
    the free tier it must stay under about 9,000 tokens or the API rejects it.
    So we walk the chunks in order, adding each to the current batch until one
    more would break either rule, then start a fresh batch.

    Returns a list of (start, end) index pairs. Keeping the order untouched
    matters, because row N of the saved embeddings must line up with chunk N.
    """
    batches = []
    start = 0
    tokens = 0

    for i, count in enumerate(token_counts):
        too_many_tokens = tokens + count > TOKENS_PER_REQUEST
        too_many_texts = i - start >= MAX_TEXTS_PER_REQUEST

        # i > start stops us making an empty batch when a single oversized
        # chunk is bigger than the whole budget on its own.
        if (too_many_tokens or too_many_texts) and i > start:
            batches.append((start, i))
            start = i
            tokens = 0

        tokens += count

    if start < len(texts):
        batches.append((start, len(texts)))

    return batches


def embed_batch(client, texts):
    """
    Send one batch of texts to Voyage and get their embeddings back.

    Note `input_type="document"`. Voyage wants to know whether a text is
    something you are storing (a "document") or something you are searching
    with (a "query"). It adds a different internal hint to each, and telling it
    the truth measurably improves results. search.py passes "query" for exactly
    this reason.

    The retry loop waits longer after each failure. That matters most for rate
    limits, which clear on their own if you simply wait.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return client.embed(texts, model=MODEL, input_type="document").embeddings
        except Exception as error:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 45 * (attempt + 1)  # 45s, 90s, 135s, ... plenty for a rate limit
            print(f"    request failed: {str(error)[:90]}")
            print(f"    waiting {wait}s before retrying")
            time.sleep(wait)


def normalise(vectors):
    """
    Scale every vector to length 1.

    Cosine similarity measures the ANGLE between two vectors and ignores their
    length. The formula is (a . b) / (|a| * |b|). If every vector already has
    length 1 then both divisors are 1, and the formula collapses to a plain dot
    product. Doing this once here makes every future search a single fast
    multiplication.
    """
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1  # guard against dividing by zero
    return vectors / lengths


def main():
    # load_dotenv reads .env and puts the keys into the environment, so the
    # secret never has to be written into the code.
    load_dotenv()

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: VOYAGE_API_KEY is not set in your .env file.")

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    client = voyageai.Client(api_key=api_key)

    # Voyage's tokeniser runs on your own machine, so counting is free and fast.
    print(f"Loaded {len(texts)} chunks. Counting tokens...")
    token_counts = [client.count_tokens([t], model=MODEL) for t in texts]
    print(f"About {sum(token_counts):,} tokens to embed.")

    # --- Don't pay twice ---
    # Embedding costs money, so if a finished embeddings.npy already matches
    # this chunks.json, stop rather than silently spending again. Deleting the
    # file is the deliberate way to force a rebuild.
    if OUTPUT_FILE.is_file():
        existing = np.load(OUTPUT_FILE)
        if existing.shape[0] == len(texts):
            print(f"\n{OUTPUT_FILE} already holds all {len(texts)} embeddings.")
            print("Nothing to do. Delete that file if you want to rebuild it.")
            return
        print(f"{OUTPUT_FILE} has {existing.shape[0]} rows but there are "
              f"{len(texts)} chunks, so it is out of date. Rebuilding.")

    # --- Resume from a previous interrupted run, if there is one ---
    done = []
    if PARTIAL_FILE.is_file():
        done = list(np.load(PARTIAL_FILE))
        print(f"Found saved progress: {len(done)} chunks already embedded.")

    batches = [b for b in plan_batches(texts, token_counts) if b[1] > len(done)]

    if not batches:
        print("Everything is already embedded.")
    else:
        minutes = len(batches) * SECONDS_BETWEEN / 60
        print(f"{len(batches)} requests still to send.", end=" ")
        print(f"Roughly {minutes:.0f} minutes." if minutes >= 1 else "Should be quick.")
        print()

    for n, (start, end) in enumerate(batches, start=1):
        # A resumed run may land mid-batch, so skip anything already done.
        start = max(start, len(done))
        if start >= end:
            continue

        done.extend(embed_batch(client, texts[start:end]))

        # Save after every single request. If the run dies now, nothing is lost.
        np.save(PARTIAL_FILE, np.array(done, dtype=np.float32))
        print(f"  [{n}/{len(batches)}] embedded {len(done)}/{len(texts)} chunks")

        # Pause before the next request so we stay under the rate limit.
        # No need to wait after the final one.
        if SECONDS_BETWEEN and n < len(batches):
            time.sleep(SECONDS_BETWEEN)

    vectors = normalise(np.array(done, dtype=np.float32))
    np.save(OUTPUT_FILE, vectors)

    # The run finished, so the resume file has done its job.
    PARTIAL_FILE.unlink(missing_ok=True)

    print(f"\nSaved {vectors.shape[0]} embeddings of {vectors.shape[1]} numbers each")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nRow 0 of embeddings.npy is the vector for chunk 0 of chunks.json,")
    print("row 1 matches chunk 1, and so on. search.py depends on that order,")
    print("so if your PDFs change, rerun ingest.py and embed.py together.")


if __name__ == "__main__":
    main()
