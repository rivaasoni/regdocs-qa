"""
ingest.py - Phase 2 of RegDocs Q&A

What this file does, in one sentence:
    It reads every PDF in the docs/ folder, turns each page into plain text,
    cuts that text into small overlapping pieces ("chunks"), and saves all the
    chunks into a single file called chunks.json.

Why chunks?
    Later phases will turn each chunk into a vector (a list of numbers) so we
    can search for the chunks most relevant to a question. Whole documents are
    too big to search well, so we work with small pieces instead.

Run it with:
    ./.venv/bin/python ingest.py
"""

import json
from pathlib import Path

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Settings - change these numbers if you want different sized chunks.
# ---------------------------------------------------------------------------

DOCS_FOLDER = Path("docs")        # where the PDFs live
OUTPUT_FILE = Path("chunks.json")  # where we save the result

CHUNK_SIZE = 400   # how many words go into one chunk
OVERLAP = 50       # how many words each chunk repeats from the previous one

# Why overlap? A sentence might sit right on the boundary between two chunks.
# Repeating the last 50 words at the start of the next chunk means an idea that
# straddles the boundary still appears whole in at least one chunk.


def read_pdf_pages(pdf_path):
    """
    Open one PDF and pull the text out of it, one page at a time.

    Returns a list of (page_number, page_text) pairs, where page_number starts
    at 1 (so it matches what a human sees in a PDF reader, not a computer's
    usual counting from 0).

    Pages with no readable text are skipped. That usually means the page is a
    scanned image rather than real text, which pypdf cannot read.
    """
    reader = PdfReader(pdf_path)
    pages = []

    # enumerate(..., start=1) numbers the pages 1, 2, 3, ... as we loop.
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""  # `or ""` guards against None
        text = text.strip()

        if text:  # only keep pages that actually gave us something
            pages.append((page_number, text))

    return pages


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Cut one blob of text into overlapping chunks of roughly `chunk_size` words.

    We count words, not characters, because words are a closer stand-in for how
    much meaning a chunk holds. "Roughly" because the last chunk is usually
    shorter, and we never split a word in half.

    Example with chunk_size=400 and overlap=50:
        chunk 1 = words 0   to 400
        chunk 2 = words 350 to 750   (the 50-word overlap is words 350-400)
        chunk 3 = words 700 to 1100
        ... and so on.
    """
    # .split() breaks the text on any whitespace and throws away the blanks,
    # which also tidies up the stray newlines that PDFs are full of.
    words = text.split()

    if not words:
        return []

    # Safety check: if overlap were >= chunk_size the loop below would never
    # move forward and the program would hang.
    if overlap >= chunk_size:
        raise ValueError("OVERLAP must be smaller than CHUNK_SIZE")

    # How far we slide forward each time. With 400 and 50, we move 350 words.
    step = chunk_size - overlap

    chunks = []
    start = 0

    while start < len(words):
        # Take a slice of up to chunk_size words, then glue them back into a
        # normal string with single spaces between them.
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))

        # If this slice already reached the end of the text, we are done.
        # Without this check we would add a tiny leftover chunk made only of
        # the overlap, which would be a duplicate of text we already have.
        if start + chunk_size >= len(words):
            break

        start += step

    return chunks


def chunk_one_pdf(pdf_path):
    """
    Turn a single PDF into a list of chunk dictionaries.

    Each dictionary looks like:
        {"text": "...", "source": "rule-2019.pdf", "page": 7}

    We chunk each page on its own rather than gluing the whole document
    together first. That costs us a little (a short page becomes one short
    chunk) but it buys something more valuable: every chunk knows exactly which
    page it came from, so later on we can cite a real page number in an answer.
    """
    chunks = []

    for page_number, page_text in read_pdf_pages(pdf_path):
        for chunk_text in split_into_chunks(page_text):
            chunks.append({
                "text": chunk_text,
                "source": pdf_path.name,  # just the filename, not the full path
                "page": page_number,
            })

    return chunks


def main():
    """
    The conductor. Finds the PDFs, chunks each one, saves the result, and
    prints a short report so you can see what happened.
    """
    # Make sure the folder exists before we go looking inside it.
    if not DOCS_FOLDER.is_dir():
        print(f"ERROR: no '{DOCS_FOLDER}/' folder found. Create it and put your PDFs there.")
        return

    # sorted() so the order is the same every time you run this, which makes
    # the output easy to compare between runs.
    # "*.pdf" is case sensitive, so we check both spellings.
    pdf_paths = sorted(
        set(DOCS_FOLDER.glob("*.pdf")) | set(DOCS_FOLDER.glob("*.PDF"))
    )

    if not pdf_paths:
        print(f"ERROR: no PDF files found in '{DOCS_FOLDER}/'.")
        print("Add at least one PDF to that folder, then run this script again.")
        return

    all_chunks = []

    for pdf_path in pdf_paths:
        # try/except so one broken or password-protected PDF does not stop the
        # whole run. We report it and carry on with the rest.
        try:
            chunks = chunk_one_pdf(pdf_path)
        except Exception as error:
            print(f"  SKIPPED {pdf_path.name}: could not read it ({error})")
            continue

        print(f"  {pdf_path.name}: {len(chunks)} chunks")

        if not chunks:
            print("    (no readable text - this PDF may be scanned images)")

        all_chunks.extend(chunks)  # extend adds every item; append would nest

    if not all_chunks:
        print("\nNo text could be extracted from any PDF. Nothing was saved.")
        return

    # Write everything to disk as JSON.
    # indent=2 makes the file readable if you open it in an editor.
    # ensure_ascii=False keeps accented characters and symbols as themselves
    # instead of turning them into escape codes like é.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ---- The report ----
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nSample chunk (the first one):")
    sample = all_chunks[0]
    print(f"  source: {sample['source']}")
    print(f"  page:   {sample['page']}")
    print(f"  words:  {len(sample['text'].split())}")
    print(f"  text:   {sample['text'][:300]}...")


# This line means "only run main() if someone runs this file directly".
# If another file imports this one later, main() will not fire by surprise.
if __name__ == "__main__":
    main()
