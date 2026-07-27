# Lesson 02 Embedding Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Load the verified token artifacts, construct shifted next-token batches, and train a minimal embedding-plus-linear language model as the no-attention baseline.

**Architecture:** Add a strict dataset loader that revalidates Lesson 01 artifacts before creating tensors. Keep batching independent from the model, then implement a tiny PyTorch model with only token embeddings and a vocabulary projection. Unit tests run on CPU; the real Tiny Shakespeare acceptance run uses the available CUDA device and writes large artifacts only below ignored `artifacts/`.

**Tech Stack:** Python 3.10+, PyTorch 2+, pytest 8+, CUDA when available, JSON, Markdown

---

### Task 1: Declare the PyTorch training dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

**Step 1: Add the dependency**

Add:

```toml
dependencies = ["torch>=2.0"]

[project.optional-dependencies]
test = ["pytest>=8"]
```

**Step 2: Refresh the editable install**

Run:

```powershell
python -m pip install -e .
```

Expected: the existing CUDA-enabled PyTorch installation satisfies the requirement; no unrelated package is downgraded.

### Task 2: Specify dataset loading and batch semantics with failing tests

**Files:**
- Create: `tests/python/test_training_dataset.py`
- Create: `training/nanogpt_nspire/training_dataset.py`

**Step 1: Write the artifact-loader tests**

Use `prepare_dataset` to create a small fixture, then require:

```python
dataset = load_token_dataset(output_dir)
assert dataset.vocabulary == ("\n", "a", "b")
assert dataset.train.dtype == torch.long
assert dataset.validation.dtype == torch.long
```

Tampering with either token file must raise `DatasetError` before a tensor is returned.

**Step 2: Write batch-shift tests**

For a known token tensor, require:

```python
x, y = make_batch(tokens, batch_size=3, block_size=4, generator=generator)
assert x.shape == (3, 4)
assert y.shape == (3, 4)
assert torch.equal(y[:, :-1], x[:, 1:])
```

Repeating with generators initialized from the same seed must return the same batch. Invalid batch size, block size, or a token stream shorter than `block_size + 1` must fail explicitly.

**Step 3: Run the new tests to verify failure**

Run:

```powershell
python -m pytest tests/python/test_training_dataset.py -q
```

Expected: collection fails because the new loader module does not exist.

### Task 3: Implement the verified tensor loader and batch sampler

**Files:**
- Create: `training/nanogpt_nspire/training_dataset.py`
- Test: `tests/python/test_training_dataset.py`

**Step 1: Implement the dataset value object**

```python
@dataclass(frozen=True)
class TokenDataset:
    train: torch.Tensor
    validation: torch.Tensor
    vocabulary: tuple[str, ...]
    manifest: dict[str, object]
```

**Step 2: Implement strict loading**

`load_token_dataset(data_dir)` must verify:

- `manifest.json` parses and has schema version 1;
- dtype is exactly `uint16-le`;
- vocabulary length matches `vocab_size`;
- each token file has the manifest byte length and SHA-256;
- each file length is even and equals token count times two;
- every decoded token ID is inside the vocabulary.

Decode through `array("H")`, correcting byte order on a big-endian Host, then convert to `torch.long`.

**Step 3: Implement deterministic batching**

`make_batch` chooses random start positions with a caller-owned CPU `torch.Generator`, slices `block_size + 1` consecutive tokens, and returns:

```text
x = window[:-1]
y = window[1:]
```

Only the finished batch moves to the requested training device.

**Step 4: Run the tests**

Run:

```powershell
python -m pytest tests/python/test_training_dataset.py -q
```

Expected: all dataset and batching tests pass.

### Task 4: Specify the no-attention language model with failing tests

**Files:**
- Create: `tests/python/test_embedding_lm.py`
- Create: `training/nanogpt_nspire/models/__init__.py`
- Create: `training/nanogpt_nspire/models/embedding_lm.py`

**Step 1: Write shape and loss tests**

For `vocab_size=5`, `embedding_dim=8`, require:

```python
logits, loss = model(token_ids, targets)
assert logits.shape == (2, 3, 5)
assert loss.ndim == 0
assert torch.isfinite(loss)
assert model.parameter_count == 2 * 5 * 8
```

Invalid tensor rank, dtype, shape, or token IDs must raise a clear error.

**Step 2: Write the learnability test**

Train the model on a repeated `abab...` transition. The final loss must be substantially below its initial loss. This proves gradients pass through embedding, linear projection, and cross-entropy.

**Step 3: Run to verify failure**

Run:

```powershell
python -m pytest tests/python/test_embedding_lm.py -q
```

Expected: collection fails because the model module does not exist.

### Task 5: Implement the embedding language model

**Files:**
- Create: `training/nanogpt_nspire/models/__init__.py`
- Create: `training/nanogpt_nspire/models/embedding_lm.py`
- Test: `tests/python/test_embedding_lm.py`

**Step 1: Implement the model**

The model contains only:

```python
self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
```

Forward data flow:

```text
(B, T) token IDs
  -> embedding
(B, T, C)
  -> linear vocabulary head
(B, T, V) logits
  -> flattened cross-entropy against targets
scalar loss
```

Do not add position embeddings, MLP, residual connections, LayerNorm, or attention.

**Step 2: Run model tests**

Run:

```powershell
python -m pytest tests/python/test_embedding_lm.py -q
```

Expected: all shape, validation, and learnability tests pass.

### Task 6: Add a reproducible Lesson 02 training command

**Files:**
- Create: `training/nanogpt_nspire/lesson02_train.py`
- Create: `tests/python/test_lesson02_train.py`

**Step 1: Test configuration and evaluation helpers**

Require:

- `resolve_device("auto")` selects CUDA when available and CPU otherwise;
- evaluation recreates its generator from a fixed seed so initial/final loss use identical validation windows;
- bits-per-character equals `loss / log(2)`;
- invalid training arguments fail before allocation.

**Step 2: Implement the training loop**

The CLI must accept:

```text
--data-dir
--output-dir
--device
--seed
--steps
--batch-size
--block-size
--embedding-dim
--learning-rate
--eval-batches
--sample-tokens
--source-commit
```

Use AdamW with zero weight decay. Record uniform random loss `log(vocab_size)`, initial validation loss, final validation loss, BPC, parameters, environment, dataset hashes, fixed-seed sample, checkpoint size, and checkpoint SHA-256.

**Step 3: Save only ignored artifacts**

Write:

```text
artifacts/lesson02/
├── embedding_lm.pt
└── run.json
```

The committed experiment summary will be created only after the implementation commit exists, so it can cite the exact source commit.

**Step 4: Run focused and full tests**

Run:

```powershell
python -m pytest tests/python/test_lesson02_train.py -q
python -m pytest -q
```

Expected: all Lesson 01 and Lesson 02 tests pass.

### Task 7: Write the Chinese lesson and commit the implementation

**Files:**
- Create: `docs/lessons/02-batches-embeddings-and-loss.md`
- Modify: `README.md`

**Step 1: Explain the concepts**

Explain:

- context windows and shifted targets;
- batch, sequence, embedding, and vocabulary dimensions;
- embedding as a trainable lookup table;
- logits versus probabilities;
- softmax inside cross-entropy;
- negative log-likelihood and bits-per-character;
- why the baseline cannot use more than the current character.

**Step 2: Explain every public function and failure case**

Connect the Python tensor shapes to the later C arrays and Nspire memory budget.

**Step 3: Verify and commit**

Run:

```powershell
python -m pytest -q
python -m compileall -q training
git diff --check
git add pyproject.toml README.md training tests docs
git commit -m "feat: add embedding language model baseline"
```

Expected: a clean implementation commit that does not contain generated artifacts.

### Task 8: Run and record the real CUDA experiment

**Files:**
- Create: `experiments/lesson02-embedding-baseline.json`
- Modify: `docs/lessons/02-batches-embeddings-and-loss.md`

**Step 1: Train from the committed implementation**

Run:

```powershell
python -m nanogpt_nspire.lesson02_train `
  --data-dir artifacts/data/tinyshakespeare `
  --output-dir artifacts/lesson02 `
  --device auto `
  --seed 1337 `
  --steps 1000 `
  --batch-size 64 `
  --block-size 64 `
  --embedding-dim 32 `
  --learning-rate 0.05 `
  --eval-batches 50 `
  --sample-tokens 300 `
  --source-commit <implementation-commit>
```

Expected:

- resolved device is CUDA on the current RTX 5080 Laptop GPU;
- final validation loss and BPC are below their initial values;
- checkpoint and run JSON are created below ignored `artifacts/`.

**Step 2: Independently verify the run**

Recompute the checkpoint hash and size, confirm dataset hashes, load the checkpoint, and run a fixed-seed sample. Do not call the baseline a Transformer: it has no attention.

**Step 3: Record observed evidence**

Copy only bounded metrics and a short sample into:

- `experiments/lesson02-embedding-baseline.json`;
- the lesson's observed-results section.

The summary must cite the implementation commit, dataset source hash, config, environment, losses, BPC, parameter count, checkpoint hash, and measured duration.

**Step 4: Commit, push, and verify**

```powershell
git add experiments/lesson02-embedding-baseline.json `
  docs/lessons/02-batches-embeddings-and-loss.md
git commit -m "docs: record lesson 02 baseline experiment"
git push origin main
python -m pytest -q
git status --short --branch
git ls-remote origin refs/heads/main
```

Expected: tests pass; local `main`, `origin/main`, and GitHub remote agree; the worktree is clean.
