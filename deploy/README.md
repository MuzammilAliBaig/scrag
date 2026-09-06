# Deployment configuration

| File | Host | Notes |
|---|---|---|
| `render.yaml` | Render free tier | Two web services from one Dockerfile |
| `spaces/` | Hugging Face Spaces | Single container, Streamlit on port 7860 |

Neither file contains a secret, and neither ever should. `ANTHROPIC_API_KEY` is set in the host
dashboard and injected at run time. A key committed here is a leaked key, and a key baked into an
image layer is leaked even if a later layer deletes it.

Full steps: `../DEPLOY.md`.
