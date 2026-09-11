"""
search.py - Phase 3b of RegDocs Q&A

What this file does, in one sentence:
    It takes a question you type, finds the chunks whose meaning is closest to
    it, and prints them with the document and page they came from.

This does NOT answer your question in words. It only finds the relevant
passages. Handing those passages to Claude to write an actual answer is
Phase 4, which lives in ask.py.

Run it with:
    conda run -n regdocs python search.py "who does EFTA protect?"
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv


CHUNKS_FILE = Path("chunks.json")
EMBEDDINGS_FILE = Path("embeddings.npy")

MODEL = "voyage-4"  # must be the same model embed.py used, or the numbers
                    # will not be comparable at all
TOP_N = 3           # how many chunks to show. Phase 4 will also use 3.


def load_everything():
    """
    Load the chunks and their embeddings, and check the two still match.

    Row 0 of the embeddings belongs to chunk 0, row 1 to chunk 1, and so on.
    Nothing enforces that except the order both files were written in, so if
    the counts disagree we stop immediately. A silent mismatch here would be
    nasty: every search would return confidently wrong passages.
    """
    if not CHUNKS_FILE.is_file() or not EMBEDDINGS_FILE.is_file():
        raise SystemExit(
            "ERROR: chunks.json or embeddings.npy is missing.\n"
            "Run ingest.py and then embed.py first."
        )

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    vectors = np.load(EMBEDDINGS_FILE)

    if len(chunks) != vectors.shape[0]:
        raise SystemExit(
            f"ERROR: {len(chunks)} chunks but {vectors.shape[0]} embeddings.\n"
            "These files are out of sync. Rerun ingest.py and then embed.py."
        )

    return chunks, vectors


def find_best_chunks(question, chunks, vectors, client, top_n=TOP_N):
    """
    Score every chunk against the question and return the best ones.

    Note `input_type="query"` here, where embed.py used "document". Voyage
    treats a question and a passage differently, and matching that to reality
    gives better results.

    The maths is three lines and worth understanding:

    1. Embed the question, then scale it to length 1, exactly as embed.py did
       to every chunk.
    2. `vectors @ q` is a dot product of the question against all 2162 chunks
       in one go. Because everything has length 1, each dot product IS the
       cosine similarity: 1.0 means identical in meaning, 0.0 means unrelated.
       numpy does all 2162 comparisons in one fast operation rather than a
       Python loop, which is the whole reason this feels instant.
    3. argsort sorts and hands back POSITIONS rather than values, which is what
       we need in order to look the original chunks back up. It sorts smallest
       first, so [::-1] flips it to put the best matches at the front.
    """
    q = np.array(client.embed([question], model=MODEL, input_type="query").embeddings[0],
                 dtype=np.float32)
    q = q / np.linalg.norm(q)

    scores = vectors @ q
    best_positions = np.argsort(scores)[::-1][:top_n]

    return [(chunks[i], float(scores[i])) for i in best_positions]


def main():
    # sys.argv holds the words typed after the filename on the command line.
    # We join them so quoting the question is optional.
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python search.py "your question here"')
    question = " ".join(sys.argv[1:])

    load_dotenv()
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: VOYAGE_API_KEY is not set in your .env file.")

    chunks, vectors = load_everything()
    client = voyageai.Client(api_key=api_key)

    results = find_best_chunks(question, chunks, vectors, client)

    print(f'\nQuestion: "{question}"')
    print(f"Searched {len(chunks)} chunks. Top {len(results)} matches:\n")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"--- {rank}. score {score:.3f} | {chunk['source']} page {chunk['page']} ---")
        print(chunk["text"][:500].strip() + "...\n")


if __name__ == "__main__":
    main()
