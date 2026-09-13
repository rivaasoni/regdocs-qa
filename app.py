"""
app.py - Phase 5, Part B

A small web interface for RegDocs Q&A, so you can ask questions without
touching a terminal.

Streamlit works differently from a normal script, and the difference is worth
understanding before you read on. Streamlit re-runs this entire file from top
to bottom every time you interact with the page: typing in a box, clicking a
button, anything. That would mean reloading 8MB of embeddings on every
keystroke, which is why the loading function below is cached.

Run it with:
    conda run -n regdocs streamlit run app.py

Then open the address it prints, usually http://localhost:8501
"""

import os

import streamlit as st
from dotenv import load_dotenv

import anthropic
import voyageai

from search import load_everything, find_best_chunks
from ask import build_context, ask_claude, TOP_N


# st.set_page_config must be the first Streamlit call in the file.
st.set_page_config(page_title="RegDocs Q&A", page_icon="📄", layout="centered")

load_dotenv()


# The @st.cache_resource decorator tells Streamlit to run this function once
# and reuse the result on every later rerun. Without it, every interaction
# would reload the embeddings file and rebuild the API clients, making the app
# painfully slow. "resource" is the right cache for things like open clients
# and big arrays, as opposed to st.cache_data which is for plain values.
@st.cache_resource
def load_pipeline():
    """Load the chunks, the embeddings, and both API clients exactly once."""
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not voyage_key or not anthropic_key:
        return None  # the caller shows a friendly message

    chunks, vectors = load_everything()
    return {
        "chunks": chunks,
        "vectors": vectors,
        "voyage": voyageai.Client(api_key=voyage_key),
        "claude": anthropic.Anthropic(api_key=anthropic_key),
    }


st.title("📄 RegDocs Q&A")
st.caption(
    "Ask a question about the loaded regulatory documents. "
    "Every answer cites the document and page it came from, "
    "and says so when the documents do not contain the answer."
)

pipeline = load_pipeline()

if pipeline is None:
    st.error(
        "Missing API keys. Create a `.env` file in the project root containing "
        "`ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`, then restart the app."
    )
    st.stop()  # nothing below this line runs

# Show what is actually loaded, so it is obvious which documents can be asked
# about. Guessing at the corpus is a common frustration with tools like this.
sources = sorted({c["source"] for c in pipeline["chunks"]})
with st.sidebar:
    st.subheader("Loaded documents")
    for name in sources:
        pages = len({c["page"] for c in pipeline["chunks"] if c["source"] == name})
        st.markdown(f"**{name}**  \n{pages} pages with text")
    st.divider()
    st.caption(f"{len(pipeline['chunks']):,} chunks indexed")
    st.caption(
        "A free Voyage account allows 3 questions per minute. "
        "Asking faster will pause and retry."
    )

question = st.text_input(
    "Your question",
    placeholder="How long does a consumer have to report an error?",
)

if st.button("Ask", type="primary") and question.strip():
    # st.spinner shows a message while the slow work happens, so the page does
    # not just sit there looking broken.
    try:
        with st.spinner("Searching the documents..."):
            results = find_best_chunks(
                question, pipeline["chunks"], pipeline["vectors"],
                pipeline["voyage"], top_n=TOP_N,
            )

        with st.spinner("Reading the passages and writing an answer..."):
            answer = ask_claude(
                pipeline["claude"], question, build_context(results)
            )
    except SystemExit as stop:
        # search.py raises SystemExit with a readable message when Voyage keeps
        # rate limiting us. In a terminal that prints and exits; in a web app we
        # need to catch it and show it, or the page would die silently.
        st.warning(str(stop))
        st.stop()

    st.subheader("Answer")

    # A refusal is worth styling differently, so it reads as a deliberate
    # outcome rather than as a broken answer.
    if answer.strip().lower().startswith("not found in the documents"):
        st.info(
            "**Not found in the documents.** The loaded documents do not "
            "appear to contain an answer to this question."
        )
    else:
        st.markdown(answer)

    st.subheader("Sources")
    st.caption(
        "These are the passages given to Claude, most relevant first. "
        "The score is cosine similarity, where 1.0 would be an exact match "
        "in meaning. Genuine matches here usually score above 0.5."
    )

    for rank, (chunk, score) in enumerate(results, start=1):
        with st.expander(
            f"{rank}. {chunk['source']} — page {chunk['page']}  ·  score {score:.3f}"
        ):
            # st.progress wants a value between 0 and 1, which a cosine
            # similarity already is, so it doubles as a little score bar.
            st.progress(min(max(score, 0.0), 1.0))
            st.write(chunk["text"])
