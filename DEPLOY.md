# Deploying SCRAG

Reproducible from a clean clone. Nothing below depends on state in a working directory, and no step
puts a secret in the image or in git.

---

## 0. The one rule

**`ANTHROPIC_API_KEY` never enters the image.** It is set in the host's environment or secrets
dashboard and injected at run time.

A key baked into a layer is a leaked key **even if a later layer deletes it** — every layer is
retrievable from a pushed image. There is no `ARG ANTHROPIC_API_KEY` in the Dockerfile and there
must never be one.

`.env` is excluded by `.dockerignore`; this is verified, not assumed:

```bash
docker build -q -f - . <<'EOF'
FROM alpine
COPY . /ctx
RUN test ! -f /ctx/.env || (echo "LEAK: .env in build context" && exit 1)
EOF
```

---

## 1. Build and run locally

```bash
git clone <repo> scrag && cd scrag
docker build -t scrag:latest .
```

The build takes roughly 5–15 minutes on a cold cache. Most of it is the CPU-only torch wheel and
pre-warming the two public model checkpoints. Neither is downloaded again at run time —
`HF_HUB_OFFLINE=1` is set in the runtime stage precisely so a missing weight fails loudly at build
rather than silently at first request.

```bash
# PowerShell
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$env:ANTHROPIC_API_KEY scrag:latest
# bash
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" scrag:latest
```

Then:

```bash
curl localhost:8000/health     # index size, model readiness
curl localhost:8000/ready      # 200 when it can serve; 503 while warming
```

### With compose

```bash
docker compose up --build      # http://localhost:8000
```

---

## 2. What is and is not in the image

| In | Why |
|---|---|
| CPU-only torch, faiss, transformers | the pipeline |
| `BAAI/bge-small-en-v1.5` | Module A embeddings, pre-warmed at build |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Module D verifier, pre-warmed at build |
| Demo corpus + an index **built during the image build** | so the first query is not a cold rebuild |

| Out | Why |
|---|---|
| `ANTHROPIC_API_KEY` | see rule 0 |
| The Module B evaluator checkpoint | not a public artifact; see below |
| The evaluation corpus and its FAISS index | ~500 MB, and irrelevant at run time |
| `eval/results/`, ALCE data, embeddings cache | not needed to serve |

### The Module B checkpoint

`data/models/evaluator/` is produced by `python -m train.finetune_evaluator` and is **not** in the
image. Without it the pipeline still runs, but Module B raises rather than silently grading with an
untrained head — deliberate, because an untrained 3-class head emits confident nonsense.

Two options:

```bash
# mount it (docker-compose already does this, read-only)
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -v "$(pwd)/data/models:/home/scrag/app/data/models:ro" \
  scrag:latest
```

or push it to a private HF repo and pull it in a build stage. If you deploy without it, turn Module
B off with the toggle under the composer, so the failure is a deliberate configuration rather than
a crash.

---

## 3. Hosting the backend - read this first

**Hugging Face Spaces is no longer free for this project.** Creating a Docker Space now returns:

```
402 Payment Required
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a PRO subscription.
```

Verified 2026-09-07 with `hf repo create scrag --repo-type space --space-sdk docker`. Only *static*
Spaces remain free, which cannot run this backend. The instructions below still work, but they need
a PRO subscription.

**Render's free tier will not fit either.** Free instances cap at 512 MB RAM, and this stack loads
torch plus two transformer checkpoints. The image is 3.6 GB.

So there is currently **no free host** for the backend. Honest options:

| Option | Cost | Notes |
|---|---|---|
| HF Spaces + PRO | paid | `deploy/spaces/` is ready; smallest change |
| Render / Railway / Fly paid tier | paid | needs >= 2 GB RAM |
| Run locally, expose with a tunnel | free | `uvicorn app.api:app` + cloudflared/ngrok; fine for a viva |
| Frontend only on Vercel | free | the page deploys; upload and ask need a backend |

**The frontend has no such problem.** It is a single 48 KB static file and Vercel hosts it free -
see section 3a.

### 3a. Streamlit Community Cloud (free, and the backend works)

**This is the recommended deployment.** The Streamlit app imports the pipeline in-process, so there
is no separate backend to host - which is exactly what made every other free option fail.

1. <https://share.streamlit.io> -> **New app** -> pick the repository
2. Main file path: `app/ui.py`
3. **Advanced settings -> Secrets**:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy.

Free instances get roughly **1 GB of RAM**. torch plus three checkpoints is tight, so if the app is
killed for memory, switch Modules **B** and **D** off in the sidebar: Module A plus generation fits
comfortably and the app degrades to cited-but-unverified answers rather than crashing.

The Module B checkpoint is not in git (`data/models/` is ignored), so **B is unavailable on a fresh
deploy** until you publish it to a private HF repo and pull it at startup, or commit it.

### 3z. Vercel - not viable, recorded for the next person

Vercel can host a static page, but not this backend: **914 MB of dependencies against a 250 MB
serverless cap**, no persistent filesystem for the FAISS index, and a 531 MB torch import per cold
start. A static frontend there would still need the Python service hosted somewhere else, which is
the problem Streamlit Cloud removes entirely.

Two things that cost time when we tried it, worth knowing: Deployment Protection is **on by
default** and redirects every visitor to a Vercel login, and it cannot be disabled from the CLI -
`vercel` CLI tokens are not authorised against the REST API, so even `GET /v9/projects/...` returns
403. It is a dashboard setting.

### 3b. Deploy to Hugging Face Spaces (needs PRO)

Spaces wants one container listening on **port 7860** running as uid 1000.

1. Create a Space: **New Space → SDK: Docker → Blank**.
2. Copy `deploy/spaces/README.md` to the Space repo root as `README.md` — Spaces reads the YAML
   header (`sdk: docker`, `app_port: 7860`) from there.
3. Copy `deploy/spaces/Dockerfile` to the Space repo root as `Dockerfile`, or build on top of
   `scrag:latest` as that file shows.
4. Push the repository.
5. **Settings → Variables and secrets → New secret**: `ANTHROPIC_API_KEY`.
6. Wait for the build. First load is slow; the UI shows a warming banner rather than hanging.

Free Spaces sleep when idle and cold-start on the next visit. That is accepted — `/ready` and the
UI banner make it visible instead of confusing.

---

## 4. Deploy to Render (alternative free tier)

`deploy/render.yaml` defines a single service. The API serves the landing page at `/`, so there
is no separate frontend host to wire up.

1. **New → Blueprint**, point it at the repository.
2. Render reads `deploy/render.yaml`. `ANTHROPIC_API_KEY` is declared `sync: false`, so it is
   **not** read from the file — set it in **Environment** on the `scrag-api` service.

Free Render instances sleep after ~15 minutes idle. First request after sleep takes 30–60 seconds.

---

## 5. Verify the deployment

```bash
curl https://<your-url>/health
curl https://<your-url>/ready
curl -X POST https://<your-url>/ingest/demo
curl -X POST https://<your-url>/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How much of my tuition is refunded if I withdraw in week four?"}'
```

Expected: a 200 whose `sentences[]` each carry `citation_ids`, and `citations[]` containing the
passage text those ids resolve to.

Then ask something the corpus does not cover — *"How much does a parking permit cost?"* — and
expect a **200 with `abstained: true`** and a reason. A refusal is the system working; it is not an
error and does not return an error code.

---

## 6. Known limitations

**Generation needs credit.** If the Claude balance is exhausted the app returns a readable 503 —
*"Retrieval, grading and verification still work; only generation is blocked"* — rather than a
stack trace. Everything except generation runs locally and free.

**Module B is out of distribution on documents unlike PopQA.** It was fine-tuned on entity
questions over Wikipedia lead sections. On policy text or uploaded PDFs its grades are unreliable,
so the free pre-generation refusal rarely fires and abstention falls to Modules C, D and E. The
guarantee holds; it just costs a generation call. See the README.

**Uploads are not persisted across restarts** on free tiers with ephemeral disks. The demo index is
rebuilt into the image, so a restart returns to the demo corpus rather than to an empty app.

**No auth, no accounts, no database** — deliberately out of scope. Do not expose an instance with
your key to the open internet without putting something in front of it.
