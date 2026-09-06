# Training Module B on a free Kaggle or Colab GPU

The evaluator fine-tune is the **only** GPU job in SCRAG. Everything else runs on CPU.

`train/finetune_evaluator.py` is device-agnostic — it picks CUDA when available and falls back to
CPU otherwise, so the same file runs here and on a free T4 with no edits.

On CPU expect roughly **40–70 minutes** for 3 epochs over ~2,600 pairs. On a free T4 expect
**2–4 minutes**. If you are iterating on hyperparameters, use the GPU.

---

## What to upload

Only two things:

| What | Path in this repo | Size |
|---|---|---|
| The weakly-labeled dataset | `data/eval_labels/{train,val,test}.jsonl` | ~4 MB |
| The training script + config | `train/finetune_evaluator.py`, `config.py`, `core/types.py` | tiny |

Do **not** upload the FAISS index, the corpus, or `.env`. The fine-tune needs none of them, and
`.env` holds your API key.

---

## Kaggle

1. **Create a Dataset.** *Datasets → New Dataset →* upload the three `.jsonl` files. Call it
   `scrag-eval-labels`. It mounts at `/kaggle/input/scrag-eval-labels/`.
2. **New Notebook**, then *Settings → Accelerator → GPU T4 x2* (one is enough) and
   *Internet → On* (needed to pull the DeBERTa weights).
3. Run the cells below.

### Cell 1 — requirements

Kaggle already ships torch. Pin only what it lacks, and match the versions this project uses:

```python
!pip install -q "transformers==5.16.1" "sentencepiece==0.2.2" "protobuf==7.36.1" "tiktoken==0.14.0"
import torch; print(torch.__version__, torch.cuda.is_available())
```

`protobuf` matters more than it looks: without it, `transformers` cannot read DeBERTa's
SentencePiece model, silently falls back to a tiktoken reader, and dies with
`Error parsing line b'\x0e' in spm.model`. That error names the wrong culprit — the fix is
`protobuf`, not `tiktoken`.

### Cell 2 — the project files

Paste `config.py`, `core/types.py` and `train/finetune_evaluator.py` into the notebook working
directory, preserving the package layout:

```python
import os, pathlib
for d in ["core", "train"]:
    pathlib.Path(d).mkdir(exist_ok=True)
    pathlib.Path(d, "__init__.py").touch()
# then write the three files with %%writefile, or attach them as a Kaggle Dataset
```

### Cell 3 — train

```python
!python -m train.finetune_evaluator \
    --data /kaggle/input/scrag-eval-labels \
    --out /kaggle/working/evaluator \
    --epochs 3 --batch-size 16 --device cuda
```

### Cell 4 — export the checkpoint

```python
!cd /kaggle/working && tar -czf evaluator.tar.gz evaluator
```

Download `evaluator.tar.gz` from the notebook's *Output* pane, then unpack it locally to the path
`config.EVALUATOR.checkpoint_path` points at:

```powershell
tar -xzf evaluator.tar.gz -C data/models/
```

---

## Colab

Same script, different mount. `!nvidia-smi` to confirm a GPU is attached, then:

```python
from google.colab import files
uploaded = files.upload()          # the three .jsonl files
!mkdir -p data/eval_labels && mv *.jsonl data/eval_labels/
!pip install -q "transformers==5.16.1" "sentencepiece==0.2.2" "protobuf==7.36.1" "tiktoken==0.14.0"
!python -m train.finetune_evaluator --epochs 3 --device cuda
```

Then `files.download('data/models/evaluator/model.safetensors')` — or zip the whole directory,
since the tokenizer files are needed too.

---

## What the checkpoint directory must contain

`core/evaluator.py` refuses to load anything incomplete, so check for all of these before
committing to a benchmark run:

```
data/models/evaluator/
├── config.json              # includes id2label - the label order is read from here
├── model.safetensors
├── spm.model
├── tokenizer_config.json
├── special_tokens_map.json
└── training_meta.json       # ours: epochs, class weights, val curve
```

If `config.json` is missing, `RetrievalEvaluator` raises `FileNotFoundError` rather than falling
back to the untrained base model. That is deliberate: an untrained 3-class head emits confident,
random-looking grades, and Module B is the one component of this project whose numbers are the
contribution.

---

## Reproducibility

`config.SEED` (42) seeds Python, NumPy and torch. GPU results will still differ slightly from CPU
results because cuDNN kernels are not bit-identical — expect drift in the third decimal, not in the
conclusion. Record which device produced any number you report; `training_meta.json` stores it.
