"""
One-off helper: runs each eval question through the real pipeline so we can
draft an expected answer for it. Not part of the eval itself.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import anthropic, voyageai

from search import load_everything, find_best_chunks
from ask import build_context, ask_claude

QUESTIONS = [
    (1,  "How many days does a consumer have to report an error after receiving a periodic statement?", False),
    (2,  "How many business days does a financial institution have to investigate a notice of error?", False),
    (3,  "What is a consumer's maximum liability for an unauthorized transfer if they notify within two business days?", False),
    (4,  "What information must be included in an initial disclosure under Regulation E?", False),
    (5,  "What is a remittance transfer under the EFTA?", False),
    (6,  "When can an institution extend its error investigation to 45 days, and what must it do first?", False),
    (7,  "What must an institution do if it determines no error occurred?", False),
    (8,  "What are the main risks associated with ACH transactions for a financial institution?", False),
    (9,  "What controls should a financial institution have over third-party payment processors?", False),
    (10, "Can a preauthorized transfer be a condition of receiving credit?", False),
    (11, "What rights does a consumer have to stop payment on a preauthorized transfer, and how much notice is required?", False),
    (12, "What is an examiner supposed to review regarding error resolution procedures?", False),
    (13, "What is the current federal funds rate?", True),
    (14, "What are the capital requirements under Basel III?", True),
    (15, "How does the FDIC calculate deposit insurance premiums?", True),
]

# The free Voyage tier allows 3 requests a minute; one question costs one.
SECONDS_BETWEEN = 30

def main():
    load_dotenv()
    chunks, vectors = load_everything()
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    out = []
    for n, (qid, question, should_refuse) in enumerate(QUESTIONS):
        results = find_best_chunks(question, chunks, vectors, vo, top_n=3)
        answer = ask_claude(claude, question, build_context(results))
        out.append({
            "id": qid,
            "question": question,
            "should_refuse": should_refuse,
            "drafted_answer": answer,
            "retrieved": [
                {"source": c["source"], "page": c["page"], "score": round(s, 3)}
                for c, s in results
            ],
        })
        print(f"  [{n+1}/{len(QUESTIONS)}] q{qid} done", flush=True)
        if n < len(QUESTIONS) - 1:
            time.sleep(SECONDS_BETWEEN)

    with open("evals/drafted.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved evals/drafted.json")

if __name__ == "__main__":
    main()
