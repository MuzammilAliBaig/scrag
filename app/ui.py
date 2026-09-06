"""SCRAG demo UI.

    streamlit run app/ui.py

Talks to the FastAPI service at SCRAG_API_URL (default http://127.0.0.1:8000).

Two things on this screen matter more than everything else, and the layout is
built around them:

1. **Every sentence carries a citation you can click open** to see the exact
   passage that supports it. That is what makes the claim "verified" legible to
   someone watching rather than a word in a slide.
2. **A refusal looks like a decision, not a bug.** An empty grey box reads as a
   crash; a clear statement that the sources do not support an answer, with the
   reason that triggered it, reads as the product feature it is.

Per-sentence badges (passed / repaired) surface the machinery deliberately. A
visibly repaired sentence is the most persuasive thing in the demo, so it is
shown rather than smoothed over.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("SCRAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 180

st.set_page_config(page_title="SCRAG", page_icon="=", layout="wide")


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def api_get(path: str):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=15)
        return response.status_code, response.json()
    except requests.exceptions.RequestException as exc:
        return None, {"detail": str(exc)}


def api_post(path: str, **kwargs):
    try:
        response = requests.post(f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"detail": response.text[:500]}
    except requests.exceptions.RequestException as exc:
        return None, {"detail": str(exc)}


# --------------------------------------------------------------------------
# Sidebar: service status and ingest
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Service")
    status, health = api_get("/health")

    if status is None:
        st.error(
            "Cannot reach the API.\n\n"
            f"Expected it at `{API_URL}`.\n\n"
            "Start it with:\n```\nuvicorn app.api:app --reload\n```"
        )
        st.stop()

    if health.get("index_ready"):
        st.success(
            f"{health['index_chunks']:,} chunks from "
            f"{health['index_documents']:,} documents"
        )
    else:
        st.warning("Index is empty. Upload documents or load the demo corpus.")

    st.caption(f"generator `{health.get('generator_model')}`")
    st.caption(f"embeddings `{health.get('embedding_model')}`")
    st.caption(f"verifier `{health.get('verifier_checkpoint')}`")

    if not health.get("api_key_configured"):
        st.error("No API key on the server. Generation is unavailable.")
    if not health.get("evaluator_checkpoint_present"):
        st.warning("Module B checkpoint missing - grading will fail.")
    if not health.get("verifier_threshold_calibrated"):
        st.warning("Verifier threshold is uncalibrated.")

    st.caption(
        "Module B was trained on PopQA entity questions over Wikipedia lead "
        "sections. On documents unlike those - policy text, contracts, uploaded "
        "PDFs - its grades are out of distribution and it rarely refuses before "
        "generation. Abstention then rests on Modules C, D and E, which still "
        "work. Measured, not assumed: see README."
    )

    st.divider()
    st.markdown("### Add documents")
    uploads = st.file_uploader(
        "PDF, .txt or .md", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploads and st.button("Index uploaded files", use_container_width=True):
        files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                 for f in uploads]
        with st.spinner("Chunking and embedding..."):
            code, payload = api_post("/ingest", files=files)
        if code == 200:
            for row in payload["files"]:
                if row["status"] == "indexed":
                    st.success(f"{row['filename']} - {row['chunks']} chunks")
                elif row["status"] == "skipped":
                    st.warning(f"{row['filename']} - {row['detail']}")
                else:
                    st.error(f"{row['filename']} - {row['detail']}")
            st.rerun()
        else:
            st.error(payload.get("detail", "ingest failed"))

    if st.button("Load demo corpus", use_container_width=True):
        with st.spinner("Indexing demo documents..."):
            code, payload = api_post("/ingest/demo")
        if code == 200:
            st.success(f"{payload['total_chunks_added']} chunks from "
                       f"{len(payload['files'])} demo documents")
            st.rerun()
        else:
            st.error(payload.get("detail", "demo ingest failed"))

    st.divider()
    st.markdown("### Pipeline")
    st.caption("Switch modules off to see what each one contributes.")
    use_evaluator = st.checkbox("B - retrieval evaluator", value=True)
    force_citations = st.checkbox("C - citation forcing", value=True)
    verify_citations = st.checkbox("D - NLI verification", value=True)
    repair_and_abstain = st.checkbox("E - repair and abstention", value=True)


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
st.title("SCRAG")
st.caption(
    "Answers only from indexed sources. Every sentence carries a citation that has been "
    "checked by entailment, and the system refuses rather than guessing."
)

question = st.text_input(
    "Question",
    placeholder="e.g. How much of my tuition is refunded if I withdraw in week four?",
)
asked = st.button("Ask", type="primary")

if asked and question.strip():
    with st.spinner("Retrieving, grading, generating, verifying..."):
        code, payload = api_post(
            "/ask",
            json={
                "question": question,
                "use_evaluator": use_evaluator,
                "force_citations": force_citations,
                "verify_citations": verify_citations,
                "repair_and_abstain": repair_and_abstain,
            },
        )

    if code is None:
        st.error(f"Could not reach the API: {payload.get('detail')}")
    elif code == 409:
        st.warning(payload.get("detail"))
    elif code == 503:
        # Quota exhaustion during a live demo. Readable, not a stack trace.
        st.error(f"**Generation unavailable.** {payload.get('detail')}")
    elif code != 200:
        st.error(payload.get("detail", f"request failed with {code}"))
    else:
        # ---------------- abstention: a decision, not an error ------------
        if payload["abstained"]:
            st.info(f"### {payload['answer']}")
            explanation = payload.get("abstain_explanation") or ""
            reason = payload.get("abstain_reason") or "unspecified"
            st.markdown(
                f"**This is a deliberate refusal, not a failure.**  \n{explanation}"
            )
            st.caption(f"trigger: `{reason}`")
            if payload.get("cost_note"):
                st.caption(f"cost: {payload['cost_note']}")

        # ---------------- answer with per-sentence citations --------------
        else:
            st.markdown("### Answer")
            citations = {c["chunk_id"]: c for c in payload["citations"]}

            for index, sentence in enumerate(payload["sentences"], 1):
                badge = {
                    "passed": ":green[verified]",
                    "repaired": ":orange[repaired]",
                    "dropped": ":red[dropped]",
                    "unverified": ":grey[unchecked]",
                }.get(sentence["status"], ":grey[unchecked]")

                score = sentence.get("entailment_score")
                score_text = f" - entailment {score:.3f}" if score is not None else ""
                st.markdown(f"**{index}.** {sentence['text']}  \n{badge}{score_text}")

                if not sentence["citation_ids"]:
                    st.caption("no citation")
                for cid in sentence["citation_ids"]:
                    chunk = citations.get(cid)
                    label = f"source: {cid}" if chunk else f"source {cid} (not returned)"
                    with st.expander(label):
                        if chunk:
                            st.write(chunk["text"])
                            st.caption(f"document `{chunk['doc_id']}`")
                        else:
                            st.caption("passage unavailable")
                st.write("")

        # ---------------- what the pipeline did ---------------------------
        with st.expander("How this answer was produced"):
            left, right = st.columns(2)
            with left:
                st.metric("Module B action", payload.get("module_b_action") or "-")
                st.metric("Retrieval rounds", payload.get("retrieval_rounds", 0))
                st.metric(
                    "Corrective retrieval",
                    "fired" if payload.get("corrective_fired") else "not needed",
                )
            with right:
                repair = payload.get("repair") or {}
                st.metric("Sentences flagged", repair.get("flagged", 0))
                st.metric("Repairs succeeded", repair.get("repair_succeeded", 0))
                st.metric("Sentences dropped", repair.get("dropped", 0))
            if repair.get("dropped"):
                st.caption(
                    f"{repair['dropped']} sentence(s) failed verification and could not be "
                    "repaired, so they were removed rather than shown to you."
                )

elif asked:
    st.warning("Type a question first.")

st.divider()
with st.expander("Try these"):
    st.markdown(
        "**The demo corpus covers these** - expect a cited answer:\n"
        "- How much of my tuition is refunded if I withdraw in week four?\n"
        "- What time does the library close on Saturday?\n"
        "- How many items can a postgraduate student borrow?\n"
        "- What happens if I arrive 40 minutes late to an examination?\n\n"
        "**It does not cover these** - expect a refusal, which is the point:\n"
        "- How much does a parking permit cost?\n"
        "- Who is the Vice-Chancellor?\n"
        "- What is the pass mark for a dissertation?"
    )
