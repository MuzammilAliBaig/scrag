# Hugging Face Spaces variant.
#
# Spaces differ from Render in three ways that matter:
#   - the container must listen on port 7860
#   - it runs as uid 1000 with $HOME writable
#   - one process per Space, so the API and UI run in the same container
#
# Build context is the REPOSITORY ROOT, not this directory:
#   docker build -f deploy/spaces/Dockerfile -t scrag-space .

FROM scrag:latest

USER root
RUN mkdir -p /home/scrag/app/data/indexes && chown -R scrag:scrag /home/scrag
USER scrag

ENV PORT=8000 \
    SCRAG_API_URL=http://127.0.0.1:8000

EXPOSE 7860

# API in the background, UI in the foreground on the port Spaces expects.
CMD ["sh", "-c", "uvicorn app.api:app --host 127.0.0.1 --port 8000 & exec streamlit run app/ui.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true"]
