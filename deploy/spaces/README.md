---
title: SCRAG
emoji: "="
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# SCRAG

A self-correcting, citation-verified RAG demo. It answers only from indexed sources, attaches a
citation to every sentence, checks each citation by entailment, and refuses when the sources do not
support an answer.

Copy this file to the root of the Space repository as `README.md` - Spaces reads the YAML header
from there.

**Configuration:** set `ANTHROPIC_API_KEY` under *Settings -> Variables and secrets -> Secrets*.
Without it, retrieval, grading and verification still work; only generation is unavailable, and the
app says so.
