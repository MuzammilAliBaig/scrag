"""SCRAG - Streamlit app.

    streamlit run app/ui.py

**This calls the pipeline in-process.** It imports core.orchestrator directly
rather than talking to FastAPI over HTTP, and that is the whole point: there is
no separate backend to host, no CORS, no second URL that can be down while the
first one is up. Deployed to Streamlit Community Cloud the backend is live
because it *is* the app.

FastAPI (app/api.py) still exists for programmatic use and is unaffected.

Two things on this screen matter more than everything else:

1. **Every sentence carries a citation you can open** to see the exact passage
   that supports it. That is what makes "verified" legible rather than a claim.
2. **A refusal reads as a decision, not a bug.** An empty box looks like a
   crash; a clear statement that the sources do not support an answer, with the
   trigger that fired, reads as the product feature it is.

Memory matters here. Streamlit Community Cloud gives about 1 GB, and torch plus
three checkpoints does not fit comfortably, so every model is loaded lazily and
cached, and Modules B and D can be switched off to stay inside the budget.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `streamlit run app/ui.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import config
from core.generator import GenerationError

st.set_page_config(
    page_title="SCRAG",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling - dark, close to the landing page it replaces
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: #0c0c0c; }
      section[data-testid="stSidebar"] { background: #111111; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .scrag-title { font-size: 2.1rem; font-weight: 600; color: #fafafa; margin-bottom: 0.1rem; }
      .scrag-sub { color: #9e9e9e; font-size: 0.95rem; margin-bottom: 1.4rem; }
      .sentence { color: #ededed; font-size: 1.02rem; line-height: 1.55; margin-bottom: 0.15rem; }
      .badge {
        display: inline-block; margin-left: 0.5em; padding: 0.05em 0.6em;
        border-radius: 999px; font-size: 0.72rem; border: 1px solid currentColor;
      }
      .badge.passed { color: #86ca8a; }
      .badge.repaired { color: #ffd400; }
      .badge.unverified { color: #9e9e9e; }
      .refusal { color: #fafafa; font-size: 1.15rem; font-weight: 500; }
      .why { color: #9e9e9e; }
      .trace { color: #7d7d7d; font-size: 0.85rem; }
      .grad-bar {
        height: 3px; border-radius: 2px; margin: 0.2rem 0 1.2rem;
        background: linear-gradient(90deg,#ffe776 0%,#ffd400 22%,#ffd000 40%,
                    #c9c93c 60%,#86ca8a 76%,#78d0cd 100%);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources. Loaded once per session, not per rerun.
# --------------------------------------------------------------------------
def app_retrieval_config() -> config.RetrievalConfig:
    """The app indexes into its own store, separate from the eval corpus.

    An upload through the UI must never pollute the corpus every measured number
    in eval/results/ was computed against.
    """
    from dataclasses import replace

    return replace(
        config.RETRIEVAL,
        index_path=config.DATA_DIR / "indexes" / "app.faiss",
        metadata_path=config.DATA_DIR / "indexes" / "app.meta.json",
        vectors_path=config.DATA_DIR / "indexes" / "app.vectors.npy",
    )


@st.cache_resource(show_spinner="Loading the retriever…")
def get_retriever():
    from core.retriever import Retriever

    r = Retriever(app_retrieval_config())
    r.ensure_index()
    return r


def evaluator_available() -> bool:
    cp = config.EVALUATOR.checkpoint_path
    return bool(cp and (cp / "config.json").exists())


def api_key_present() -> bool:
    return bool(config.has_api_key() or os.getenv("ANTHROPIC_API_KEY") or
                st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else False)


# --------------------------------------------------------------------------
# Secrets: Streamlit Community Cloud supplies them via st.secrets, and the
# anthropic SDK reads the environment, so bridge the two.
# --------------------------------------------------------------------------
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### SCRAG")
    st.caption("Answers only from indexed sources, with every citation verified.")

    retriever = get_retriever()

    if retriever.size:
        st.success(f"{retriever.size:,} chunks · {len(retriever.documents):,} documents")
    else:
        st.warning("Index is empty. Load the demo corpus or upload documents.")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error(
            "No `ANTHROPIC_API_KEY`. Retrieval, grading and verification still "
            "work; generating an answer does not."
        )
    if not evaluator_available():
        st.warning(
            "Module B checkpoint missing. Switch **B** off below, or the "
            "pipeline will refuse rather than grade with an untrained head."
        )

    st.divider()
    st.markdown("#### Add documents")
    uploads = st.file_uploader(
        "PDF, .txt or .md", type=["pdf", "txt", "md"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    bulk = st.checkbox(
        "Bulk mode (coarser chunks)", value=False,
        help="Fewer, larger chunks. Embedding scales with tokens, so this saves "
             "about 1.2x, not the 2.6x the chunk count suggests.",
    )

    if uploads and st.button("Index uploaded files", use_container_width=True, type="primary"):
        from app.api import extract_text

        docs, report = {}, []
        for f in uploads:
            try:
                text = extract_text(f.name, f.getvalue())
                docs[f.name.rsplit(".", 1)[0]] = text
                report.append(("ok", f.name, f"{len(text):,} characters"))
            except ValueError as exc:
                report.append(("bad", f.name, str(exc)))

        if docs:
            with st.spinner(f"Chunking and embedding {len(docs)} document(s)…"):
                added = retriever.add_documents(docs, bulk=bulk)
            for kind, name, detail in report:
                key = name.rsplit(".", 1)[0]
                if kind == "ok":
                    st.success(f"{name} — {added.get(key, 0)} chunks")
                else:
                    st.error(f"{name} — {detail}")
            st.rerun()
        else:
            for _, name, detail in report:
                st.error(f"{name} — {detail}")

    if st.button("Load demo corpus", use_container_width=True):
        from app.demo_corpus import ingest_demo

        with st.spinner("Indexing demo documents…"):
            added = ingest_demo(retriever)
        st.success(f"{sum(added.values())} chunks from {len(added)} documents")
        st.rerun()

    st.divider()
    st.markdown("#### Pipeline")
    st.caption("Switch stages off to see what each contributes.")
    use_evaluator = st.checkbox("B · retrieval evaluator", value=evaluator_available())
    force_citations = st.checkbox("C · citation forcing", value=True)
    verify_citations = st.checkbox("D · NLI verification", value=True)
    repair_and_abstain = st.checkbox("E · repair and abstention", value=True)

    st.divider()
    st.caption(f"generator `{config.GENERATOR.model}`")
    st.caption(f"embeddings `{config.RETRIEVAL.embedding_model}`")
    st.caption(f"verifier `{config.VERIFIER.nli_model}`")
    st.caption(
        "Module B was fine-tuned on PopQA entity questions over Wikipedia lead "
        "sections. On policy text or uploaded PDFs its grades are out of "
        "distribution — measured, not assumed."
    )


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
st.markdown('<div class="scrag-title">Think clearly. Decide confidently.</div>',
            unsafe_allow_html=True)
st.markdown('<div class="grad-bar"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="scrag-sub">Every sentence carries a citation that has been checked by '
    'entailment, and the system refuses rather than guessing.</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Question",
    placeholder="Break down a decision, problem, or idea…",
    height=90,
    label_visibility="collapsed",
)
asked = st.button("Ask", type="primary")


def render(answer) -> None:
    """Render a FinalAnswer the way the pipeline actually produced it."""
    trace = answer.trace or {}

    if answer.abstained:
        st.info(f"**{answer.text}**")
        st.markdown(
            '<span class="why">This is a deliberate refusal, not a failure. '
            f'{trace.get("abstain_explanation", "")}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"trigger `{answer.abstain_reason}`"
                   + (f" · {trace['cost']}" if "cost" in trace else ""))
        return

    verdicts = {v.sentence.text: v for v in (trace.get("verdicts") or [])}
    by_id = {c.chunk_id: c for c in answer.citations}
    repaired = (trace.get("repair") or {}).get("repair_succeeded", 0)

    for sentence in answer.sentences:
        verdict = verdicts.get(sentence.text)
        if verdict is None:
            status, score = ("repaired" if repaired else "unverified"), None
        else:
            status = "passed" if verdict.supported else "unverified"
            score = verdict.entailment_score

        label = status + (f" {score:.2f}" if score is not None else "")
        st.markdown(
            f'<div class="sentence">{sentence.text}'
            f'<span class="badge {status}">{label}</span></div>',
            unsafe_allow_html=True,
        )
        for cid in sentence.citation_ids:
            chunk = by_id.get(cid)
            with st.expander(f"source · {cid}"):
                st.write(chunk.text if chunk else "passage unavailable")
                if chunk:
                    st.caption(f"document `{chunk.doc_id}`")
        if not sentence.citation_ids:
            st.caption("no citation")

    retrieval = trace.get("retrieval") or {}
    rep = trace.get("repair") or {}
    st.markdown(
        f'<div class="trace">Module B <code>{retrieval.get("action", "-")}</code> · '
        f'retrieval rounds {retrieval.get("rounds", 0)} · '
        f'corrective {"fired" if retrieval.get("corrective_fired") else "not needed"} · '
        f'flagged {rep.get("flagged", 0)} · repaired {rep.get("repair_succeeded", 0)} · '
        f'dropped {rep.get("dropped", 0)}</div>',
        unsafe_allow_html=True,
    )


if asked and question.strip():
    if retriever.size == 0:
        st.warning("The index is empty. Load the demo corpus or upload documents first.")
    else:
        from core.orchestrator import Orchestrator, PipelineFlags

        orch = Orchestrator(
            retriever=retriever,
            flags=PipelineFlags(
                use_evaluator=use_evaluator,
                force_citations=force_citations,
                verify_citations=verify_citations,
                repair_and_abstain=repair_and_abstain,
            ),
        )
        try:
            with st.spinner("Retrieving, grading, generating, verifying…"):
                answer = orch.answer(question)
            render(answer)
        except GenerationError as exc:
            message = str(exc)
            if "credit balance is too low" in message:
                st.error(
                    "**Generation unavailable.** The Claude API credit balance is "
                    "exhausted. Retrieval, grading and verification still work; only "
                    "generating an answer is blocked."
                )
            elif "ANTHROPIC_API_KEY is not set" in message:
                st.error(
                    "**No API key.** Set `ANTHROPIC_API_KEY` in the app secrets "
                    "(Streamlit Cloud) or your environment."
                )
            else:
                st.error(f"Generation failed: {message}")
        except FileNotFoundError as exc:
            st.error(f"**A model is missing.** {exc}")
elif asked:
    st.warning("Type a question first.")

with st.expander("Try these"):
    st.markdown(
        "**Covered by the demo corpus** — expect a cited answer:\n"
        "- How much of my tuition is refunded if I withdraw in week four?\n"
        "- What time does the library close on Saturday?\n"
        "- How many items can a postgraduate student borrow?\n\n"
        "**Not covered** — expect a refusal, which is the point:\n"
        "- How much does a parking permit cost?\n"
        "- Who is the Vice-Chancellor?"
    )
