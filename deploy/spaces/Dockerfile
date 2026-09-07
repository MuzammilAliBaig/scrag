# Hugging Face Spaces variant.
#
# Spaces differ from Render in three ways that matter:
#   - the container must listen on port 7860
#   - it runs as uid 1000 with $HOME writable
#   - one process per Space, which suits us: FastAPI serves the JSON API and
#     the landing page from the same app
#
# Build context is the REPOSITORY ROOT, not this directory:
#   docker build -f deploy/spaces/Dockerfile -t scrag-space .

FROM scrag:latest

USER root
RUN mkdir -p /home/scrag/app/data/indexes && chown -R scrag:scrag /home/scrag
USER scrag

ENV PORT=7860

EXPOSE 7860

# One process: FastAPI serves the landing page at / and the JSON endpoints.
CMD ["sh", "-c", "exec uvicorn app.api:app --host 0.0.0.0 --port 7860"]
