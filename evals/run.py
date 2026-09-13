"""
evals/run.py - Phase 5, Part A

What this file does, in one sentence:
    It asks the system all 15 test questions, checks each answer three ways,
    and prints a scorecard.

Why bother?
    Without this, every change you make to the project is an unverified
    opinion. Tweak the chunk size, swap the model, reword the prompt: is it
    better or worse? An eval turns that question into a number you can compare.
    Build it before you start changing things, not after.

The three things measured:

1. Retrieval hit rate - did the document we expected actually appear in the
   top 3 chunks? This tests Phase 3 on its own. If retrieval misses, no amount
   of clever prompting in Phase 4 can save the answer.

2. Answer correctness - we show Claude the expected answer and the actual
   answer and ask whether the actual one conveys the same key facts. This is
   called "LLM as judge". It is not perfect, and one honest caveat is that the
   judge is the same model family that wrote the answer, which can make it a
   soft marker. It is still far better than eyeballing 15 answers by hand.

3. Refusal accuracy - for the 3 out-of-scope questions, did it correctly say
   "Not found in the documents"? And just as importantly, did it avoid refusing
   the 12 questions it should have answered?

Run it with:
    conda run -n regdocs python evals/run.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Make the project root importable so we can reuse the real pipeline rather
# than reimplementing it. An eval that tests a copy of the code tests nothing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
import voyageai
from dotenv import load_dotenv

from search import load_everything, find_best_chunks
from ask import build_context, ask_claude


QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.json"

JUDGE_MODEL = "claude-sonnet-5"
REFUSAL_TEXT = "not found in the documents"

# A free Voyage account allows 3 requests a minute and each question costs one,
# so we wait between questions. Without this the run dies partway through.
SECONDS_BETWEEN = 30

JUDGE_PROMPT = """You are grading a question-answering system that must answer only from regulatory documents.

You will be given a question, a reference answer, and the system's actual answer.

Decide whether the actual answer conveys the same key facts as the reference
answer. Judge the substance, not the wording, and ignore differences in length,
formatting or citation style. Extra correct detail is fine. Missing a key fact,
such as a specific deadline or dollar amount, or stating something that
contradicts the reference, is not.

Reply with exactly one word on the first line, CORRECT or INCORRECT, then one
short sentence explaining why."""


def judge(client, question, expected, actual):
    """
    Ask Claude whether the actual answer matches the expected one.

    Returns (is_correct, reason). We parse only the first word, which is why
    the prompt above is so insistent about the format.
    """
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1000,
        system=JUDGE_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Reference answer:\n{expected}\n\n"
                f"Actual answer:\n{actual}"
            ),
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    first_line = text.split("\n")[0].strip().upper()
    reason = " ".join(text.split("\n")[1:]).strip() or text
    return first_line.startswith("CORRECT"), reason[:150]


def main():
    load_dotenv()
    for key in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"ERROR: {key} is not set in your .env file.")

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    chunks, vectors = load_everything()
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []

    for n, q in enumerate(questions):
        # --- run the real pipeline ---
        retrieved = find_best_chunks(q["question"], chunks, vectors, vo, top_n=3)
        answer = ask_claude(claude, q["question"], build_context(retrieved))

        sources = [c["source"] for c, _ in retrieved]
        refused = answer.strip().lower().startswith(REFUSAL_TEXT)

        # --- measure 1: did we retrieve the right document? ---
        # Meaningless for the refusal questions, which have no expected source.
        if q["expected_source"] is None:
            retrieval_hit = None
        else:
            retrieval_hit = q["expected_source"] in sources

        # --- measure 3: refusal behaviour ---
        # Correct means: refused when it should, and did not refuse otherwise.
        refusal_ok = (refused == q["should_refuse"])

        # --- measure 2: is the answer right? ---
        # For a refusal question the answer is right exactly when it refused,
        # so there is nothing for the judge to weigh up and we skip the call.
        if q["should_refuse"]:
            correct, reason = refused, "refused as expected" if refused else "failed to refuse"
        elif refused:
            correct, reason = False, "refused a question it should have answered"
        else:
            correct, reason = judge(claude, q["question"], q["expected_answer"], answer)

        results.append({
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "expected_source": q["expected_source"],
            "retrieved_sources": sources,
            "top_score": round(retrieved[0][1], 3),
            "retrieval_hit": retrieval_hit,
            "refused": refused,
            "refusal_ok": refusal_ok,
            "correct": correct,
            "reason": reason,
            "answer": answer,
        })

        print(f"  [{n + 1}/{len(questions)}] q{q['id']} done", flush=True)
        if n < len(questions) - 1:
            time.sleep(SECONDS_BETWEEN)

    report(results)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {RESULTS_FILE}")


def report(results):
    """Print the per-question table and the overall scores."""
    print("\n" + "=" * 78)
    print(f"{'id':>3}  {'type':<8} {'score':>6}  {'retr':^5} {'refuse':^6} {'answer':^7}  note")
    print("-" * 78)

    for r in results:
        retr = "-" if r["retrieval_hit"] is None else ("PASS" if r["retrieval_hit"] else "MISS")
        ref = "PASS" if r["refusal_ok"] else "FAIL"
        ans = "PASS" if r["correct"] else "FAIL"
        note = "" if r["correct"] else r["reason"][:32]
        print(f"{r['id']:>3}  {r['difficulty']:<8} {r['top_score']:>6.3f}  "
              f"{retr:^5} {ref:^6} {ans:^7}  {note}")

    # Score each measure over only the questions it applies to.
    scored = [r for r in results if r["retrieval_hit"] is not None]
    answerable = [r for r in results if not r["expected_source"] is None]
    refusals = [r for r in results if r["expected_source"] is None]

    hits = sum(1 for r in scored if r["retrieval_hit"])
    correct = sum(1 for r in results if r["correct"])
    ans_correct = sum(1 for r in answerable if r["correct"])
    ref_ok = sum(1 for r in results if r["refusal_ok"])

    print("=" * 78)
    print("OVERALL")
    print(f"  Retrieval hit rate    {hits}/{len(scored)}   "
          f"({hits / len(scored) * 100:.0f}%)  expected document in top 3")
    print(f"  Answer correctness    {ans_correct}/{len(answerable)}   "
          f"({ans_correct / len(answerable) * 100:.0f}%)  judged against reference")
    print(f"  Refusal accuracy      {ref_ok}/{len(results)}   "
          f"({ref_ok / len(results) * 100:.0f}%)  refused only when it should")
    print(f"  Overall correct       {correct}/{len(results)}   "
          f"({correct / len(results) * 100:.0f}%)  all 15 questions")

    # Score separation is worth watching: if out-of-scope questions do not score
    # clearly lower than in-scope ones, similarity alone cannot gate answering.
    in_scope = [r["top_score"] for r in results if r["expected_source"]]
    out_scope = [r["top_score"] for r in refusals]
    if out_scope:
        print(f"\n  Similarity range, in scope     {min(in_scope):.3f} - {max(in_scope):.3f}")
        print(f"  Similarity range, out of scope {min(out_scope):.3f} - {max(out_scope):.3f}")
        gap = min(in_scope) - max(out_scope)
        print(f"  Gap between them               {gap:+.3f}"
              f"{'  (overlapping, so a threshold alone is unsafe)' if gap <= 0.1 else ''}")


if __name__ == "__main__":
    main()
