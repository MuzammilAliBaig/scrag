# Reproducibility

Everything needed to regenerate every number in this project, and an honest account of what will
and will not reproduce exactly.

---

## Hardware and environment used for the recorded runs

| | |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | Intel64 Family 6 Model 141 (11th gen), 12 logical cores |
| GPU | **none** — every recorded number was produced on CPU |
| Python | **3.11.9** |

Python 3.11 is not incidental. The default interpreter on the development machine was 3.14, and
faiss / torch / sentence-transformers wheels are not dependable there. Use 3.11.

## Pinned versions

| Package | Version |
|---|---|
| torch | 2.14.0**+cpu** |
| faiss-cpu | 1.15.0 |
| transformers | 5.16.1 |
| sentence-transformers | 6.0.1 |
| numpy | 2.4.6 |
| anthropic | 1.4.0 |
| datasets | 5.0.1 |
| fastapi / uvicorn | 0.141.1 / 0.52.4 |
| streamlit | 1.63.0 |

Full sets: `requirements.txt` (development and evaluation) and `requirements-runtime.txt` (what the
Docker image installs).

**Three pins that are easy to get wrong:**

- `protobuf==7.36.1` is **required** to read DeBERTa's `spm.model`. Without it `transformers` falls
  back to a tiktoken reader and dies with ``Error parsing line b'\x0e' in spm.model`` — an error
  that names the wrong library entirely.
- `sentencepiece` and `tiktoken` come along the same tokenizer path.
- Set `HF_HUB_DISABLE_XET=1`. The Xet transfer backend stalled HuggingFace downloads twice during
  development, once silently at 0 bytes for six minutes.

## Models

| Role | Checkpoint | Provenance |
|---|---|---|
| Embeddings (A) | `BAAI/bge-small-en-v1.5` | public |
| Evaluator (B) | `microsoft/deberta-v3-small`, fine-tuned | **produced here**, `train/finetune_evaluator.py` |
| Verifier (D) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | public, off the shelf |
| Generator (C) | `claude-opus-5` | Claude API — the only paid component |

## Seeds

`config.SEED = 42` seeds Python, NumPy and torch. It fixes:

- the PopQA dev/test/distractor split (`data/splits/*.json`, **committed**),
- the Module B train/val/test split, partitioned **by question** so no question straddles splits,
- the negative sampling in the verifier calibration slice,
- the fine-tuning run.

In CI, `PYTHONHASHSEED=0` and `TOKENIZERS_PARALLELISM=false` are set as well.

---

## What reproduces exactly, and what does not

**Exactly (deterministic, CPU, no network):**
retrieval hit rate · Module B grading accuracy and F1 · citation P/R and the automatic hallucination
rate *when run against the committed response cache* · the verifier threshold sweep · abstention
trigger (b) rates.

**Approximately:**
- **The fine-tune.** CPU and GPU produce slightly different weights — cuDNN kernels are not
  bit-identical. Expect drift in the third decimal, not in the conclusion. `training_meta.json`
  records the device.
- **Anything downstream of generation.** The Claude API is not deterministic even at temperature 0.
  This is why the CI gate's thresholds carry margins sized to each metric's noise.

**Not at all, without an API key:** the Phase 1 baseline, the A–E ablation, and any fresh
generation. The on-disk response cache (`eval/results/.api_cache/`) makes the *citation* metrics
reproducible without a key, because they are recomputed from stored generations.

**One number changed during the build and is worth flagging:** the index was rebuilt from 1,612 to
**1,615 chunks** after a chunker fix (word-boundary snapping). The Phase 5 abstention measurements
were re-run against the rebuilt index and were **unchanged** at 0.8267 / 0.0267.

---

## Run commands, in order

### Free — no API key required

```bash
py -3.11 -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# Module A: fetch the corpus and build the index (~6 min, Wikipedia + CPU embedding)
python -m eval.build_index --split dev --limit 200 --distractors 800
python -m eval.check_retrieval --split dev --limit 200

# Module B: dataset, fine-tune, benchmark
python -m train.build_dataset --out data/eval_labels --split test
python -m train.finetune_evaluator --epochs 3 --batch-size 16   # ~86 min CPU, ~3 min on a free T4
python -m eval.run_evaluator_bench --split test

# Module D: calibration and verification (needs the response cache)
python -m eval.calibrate_verifier --limit 200
python -m eval.run_verifier_bench --limit 200
python -m eval.run_citation_pr --limit 200

# Module E: the free half of the abstention measurement
python -m eval.run_abstention_bench --limit 150 --no-generation

# Report artifacts
python -m docs.export_results
python -m docs.make_figures

# The CI gate
python -m eval.check_regression
```

### Paid — needs `ANTHROPIC_API_KEY`

Price every sweep before launching it. `--dry-run` uses `messages.count_tokens`.

```bash
python -m eval.run_citation_bench --dataset alce --limit 200 --dry-run
python -m eval.run_citation_bench --dataset alce --limit 200      # measured: $2.18
python -m eval.run_baseline --split dev --limit 150               # ~$2-3
python -m eval.run_abstention_bench --limit 150                   # ~$3-4
python -m eval.run_ablation --all --split dev --limit 150 --dry-run
python -m eval.run_ablation --all --split dev --limit 150         # ~$6-8, or ~$3-4 batched
python -m eval.plot_ablation
```

**Measured cost: $0.01089 per question** (claude-opus-5, prompt caching active, 200 ALCE questions).
Use that figure rather than the estimates in `tech-stack.md`, which assumed token counts.

### External data

- **PopQA** — `akariasai/PopQA` via `datasets`, downloaded automatically.
- **ALCE** — `princeton-nlp/ALCE-data`, a 450 MB tarball, **not committed**:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('princeton-nlp/ALCE-data','ALCE-data.tar',repo_type='dataset'))"
# extract into data/alce/
```

- **Corpus** — Wikipedia lead sections fetched from the MediaWiki API by `eval/build_index`. Note
  that `prop=extracts` returns **one** extract per request unless both `exlimit=max` and `exintro`
  are set; without them 19 of every 20 pages are silently dropped.

---

## Every number traces to a file

| Artifact | Produced by |
|---|---|
| `eval/results/retrieval_check.json` | `eval.check_retrieval` |
| `eval/results/evaluator_bench.json` | `eval.run_evaluator_bench` |
| `eval/results/verifier_calibration.json` | `eval.calibrate_verifier` |
| `eval/results/verifier_bench.json` | `eval.run_verifier_bench` |
| `eval/results/citation_bench.json` | `eval.run_citation_bench` |
| `eval/results/citation_pr.json` | `eval.run_citation_pr` |
| `eval/results/abstention_bench_retrieval_only.json` | `eval.run_abstention_bench --no-generation` |
| `eval/results/phase2_summary.json` | hand-assembled from the two runs above it, both disclosed |
| `docs/tables/*.csv`, `docs/figures/*.png` | `docs.export_results`, `docs.make_figures` |

`docs/tables/measured_results.csv` carries the source file for every row. Rows with an empty value
were **not measured**, and the note says why — they are never rendered as zero.
