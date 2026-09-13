"""
ask.py - Phase 4 of RegDocs Q&A

What this file does, in one sentence:
    It takes your question, finds the 3 most relevant passages from the
    regulatory PDFs, hands them to Claude, and prints an answer that cites the
    document and page each claim came from.

This is the piece that makes the whole project useful. Phase 3 could find
relevant passages, but you still had to read them yourself. Now Claude reads
them for you and writes the answer.

The most important idea here is "grounding". We do NOT ask Claude what it
knows about banking regulations. We ask it to answer using only the passages
we hand it, and to say so plainly when those passages do not contain the
answer. A confident wrong answer about a financial regulation is far worse
than an honest "I could not find that".

Run it with:
    conda run -n regdocs python ask.py "who does Regulation E protect?"
"""

import os
import sys

import anthropic
import voyageai
from dotenv import load_dotenv

# Reuse the search code from Phase 3 rather than copying it. Because search.py
# guards its own code behind `if __name__ == "__main__"`, importing it here
# does not accidentally run its command-line behaviour.
from search import load_everything, find_best_chunks


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-5"  # Anthropic's current Sonnet model
TOP_N = 3                  # how many passages to give Claude

# The system prompt sets the rules Claude must follow. Writing rules here,
# rather than in the question itself, keeps them separate from user input and
# makes them apply to every question you ever ask.
SYSTEM_PROMPT = """You are a careful assistant answering questions about financial regulations.

Follow these rules exactly:

1. Answer ONLY using the numbered context passages provided in the user's
   message. Do not use any other knowledge, even if you are confident it is
   correct.
2. Cite your source for every claim you make, using the filename and page
   given with each passage, like this: (consumer compliance handbook.pdf, p. 97).
3. If the passages do not contain the answer, reply with exactly:
   Not found in the documents
   Do not guess, and do not pad the reply with anything else.
4. Be concise. Quote the regulation's own wording where it is precise and
   matters, such as specific deadlines or dollar amounts.

A wrong answer about a regulation is worse than no answer."""


def build_context(results):
    """
    Turn the search results into one block of text for Claude to read.

    Each passage is numbered and clearly labelled with its filename and page.
    That labelling is what makes citation possible: Claude can only cite a page
    if we tell it which page the words came from in the first place.
    """
    parts = []
    for i, (chunk, score) in enumerate(results, start=1):
        parts.append(
            f"[Passage {i}] from {chunk['source']}, page {chunk['page']}:\n"
            f"{chunk['text']}"
        )
    # A blank line between passages so they do not visually run together.
    return "\n\n".join(parts)


def ask_claude(client, question, context):
    """
    Send the question and the passages to Claude, and return its answer.

    Note what we are NOT setting. There is no `temperature` argument, because
    the current Sonnet model rejects it. There is no `thinking` argument
    either, because leaving it out means Claude decides for itself how much to
    think, which is the sensible default.

    max_tokens is a ceiling on the reply length, not a target. 4000 is far more
    than a cited answer needs, which simply means we will never be cut off
    mid-sentence.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Context passages:\n\n{context}\n\n"
                f"---\n\nQuestion: {question}"
            ),
        }],
    )

    # response.content is a list of blocks, not a plain string. A reply can
    # contain thinking blocks as well as text, so we keep only the text ones
    # and join them together.
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def main():
    # sys.argv holds whatever was typed after the filename. Joining the words
    # means quoting your question is optional.
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python ask.py "your question here"')
    question = " ".join(sys.argv[1:])

    load_dotenv()

    # Check both keys up front. Failing here with a clear message is much
    # kinder than failing halfway through with a stack trace.
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not voyage_key:
        raise SystemExit("ERROR: VOYAGE_API_KEY is not set in your .env file.")
    if not anthropic_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY is not set in your .env file.")

    # Step 1: load the chunks and their embeddings from Phase 3.
    chunks, vectors = load_everything()

    # Step 2: find the passages closest in meaning to the question.
    voyage_client = voyageai.Client(api_key=voyage_key)
    results = find_best_chunks(question, chunks, vectors, voyage_client, top_n=TOP_N)

    # Step 3: hand those passages to Claude and let it write the answer.
    claude = anthropic.Anthropic(api_key=anthropic_key)
    answer = ask_claude(claude, question, build_context(results))

    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print(answer)

    # Printing the sources separately lets you go and check Claude's work.
    # This is the whole point of a citing system: it should be verifiable.
    print("\nSources given to Claude:")
    for i, (chunk, score) in enumerate(results, start=1):
        print(f"  {i}. {chunk['source']}, page {chunk['page']} (similarity {score:.3f})")


if __name__ == "__main__":
    main()
