# Deployment configuration

| File | Host | Notes |
|---|---|---|
| `render.yaml` | Render free tier | One web service; the API serves the page at `/` |
| `spaces/` | Hugging Face Spaces | Single container, FastAPI on 7860. **Needs HF PRO** - Docker Spaces are no longer free |
| `../vercel.json` | Vercel | The static landing page only; the backend cannot fit |

Neither file contains a secret, and neither ever should. `ANTHROPIC_API_KEY` is set in the host
dashboard and injected at run time. A key committed here is a leaked key, and a key baked into an
image layer is leaked even if a later layer deletes it.

Full steps: `../DEPLOY.md`.
