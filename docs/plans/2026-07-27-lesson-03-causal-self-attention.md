# Lesson 03 Causal Self-Attention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a transparent single-head causal self-attention model that can use earlier characters without reading future targets, then compare it with the Lesson 02 no-context baseline.

**Architecture:** Keep the manually computed attention path visible: learned token and position embeddings produce Q/K/V, scaled dot-product scores receive a lower-triangular mask, softmax weights mix values, and an output projection returns through one residual connection. Do not add multi-head attention, LayerNorm, MLP, dropout, or Flash Attention yet. Reuse the verified data/batch layer and extract only genuinely shared experiment helpers from Lesson 02.

**Tech Stack:** Python 3.10+, PyTorch 2+, pytest 8+, CUDA, JSON, Markdown

---

### Task 1: Specify causal attention with failing tests

**Files:**
- Create: `tests/python/test_causal_attention_lm.py`
- Create: `training/nanogpt_nspire/models/causal_attention_lm.py`
- Modify: `training/nanogpt_nspire/models/__init__.py`

**Step 1: Test attention shapes and mask**

For input shape `(B=2, T=4, C=8)`, require:

```python
output, weights = attention(x, return_weights=True)
assert output.shape == (2, 4, 8)
assert weights.shape == (2, 4, 4)
assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))
```

**Step 2: Test future isolation**

Create two sequences that are identical through position 2 and differ only afterward. Their logits at positions 0–2 must be identical within floating-point tolerance.

This test verifies behavior, not merely that a triangular buffer exists.

**Step 3: Test model shapes, limits, and parameter count**

For vocabulary `V`, block size `T`, and embedding width `C`, require:

```text
parameters = 2VC + TC + 4C²
```

The model must reject sequences longer than its block size, invalid token tensors, and mismatched targets.

**Step 4: Test context-dependent learning**

Build a toy task where the current delimiter token is identical but its target equals an earlier context token. Train the attention model and require a large loss reduction. The Lesson 02 current-token-only model cannot solve this mapping in general.

**Step 5: Run to verify failure**

Run:

```powershell
python -m pytest tests/python/test_causal_attention_lm.py -q
```

Expected: collection fails because `causal_attention_lm` does not exist.

### Task 2: Implement manual single-head causal self-attention

**Files:**
- Create: `training/nanogpt_nspire/models/causal_attention_lm.py`
- Modify: `training/nanogpt_nspire/models/__init__.py`
- Test: `tests/python/test_causal_attention_lm.py`

**Step 1: Implement Q, K, and V**

Use four bias-free `C -> C` linear layers:

```python
self.query = nn.Linear(C, C, bias=False)
self.key = nn.Linear(C, C, bias=False)
self.value = nn.Linear(C, C, bias=False)
self.output = nn.Linear(C, C, bias=False)
```

**Step 2: Implement scaled scores and mask**

```text
q, k, v: (B,T,C)
scores = q @ k.transpose(-2,-1) / sqrt(C): (B,T,T)
scores[future] = -inf
weights = softmax(scores, dim=-1)
context = weights @ v: (B,T,C)
```

Register a boolean lower-triangular mask of shape `(block_size, block_size)`. Slice it to the active sequence length.

**Step 3: Implement the learning model**

```text
token IDs (B,T)
  -> token embedding + learned position embedding
  -> x (B,T,C)
  -> x + attention_output(x)
  -> vocabulary head
  -> logits (B,T,V)
```

Use one residual addition to preserve current-token information. Do not add any other Transformer-block components.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests/python/test_causal_attention_lm.py -q
```

Expected: mask, future-isolation, shape, error, and learning tests all pass.

### Task 3: Extract shared training support without changing Lesson 02 behavior

**Files:**
- Create: `training/nanogpt_nspire/training_support.py`
- Modify: `training/nanogpt_nspire/lesson02_train.py`
- Test: `tests/python/test_lesson02_train.py`

**Step 1: Move reusable helpers**

Move or wrap:

- `resolve_device`;
- `bits_per_character`;
- `evaluate_loss`;
- device synchronization;
- SHA-256 file hashing;
- atomic JSON writing;
- bounded dataset summary.

Keep the Lesson 02 names importable from `lesson02_train.py` so its existing tests and CLI remain compatible.

**Step 2: Generalize only the model protocol**

Shared evaluation may accept any model whose forward returns `(logits, loss)`. Do not merge the lesson-specific configs, checkpoint schemas, or sampling algorithms.

**Step 3: Verify no regression**

Run:

```powershell
python -m pytest tests/python/test_lesson02_train.py -q
python -m pytest -q
```

Expected: the previously recorded Lesson 02 behavior remains tested and all tests pass.

### Task 4: Specify context-aware generation and training CLI

**Files:**
- Create: `tests/python/test_lesson03_train.py`
- Create: `training/nanogpt_nspire/lesson03_train.py`

**Step 1: Test autoregressive context cropping**

Generation must feed up to the last `block_size` generated tokens into the model, not only the final token. Fixed seed, prompt, checkpoint, and temperature must reproduce the same sequence.

**Step 2: Test configuration failure before training**

Reject non-positive model/training dimensions, invalid temperature, a requested sequence longer than data, and unavailable devices before the training loop.

**Step 3: Add a bounded CPU smoke run**

Prepare a small dataset, train for a few steps, and verify:

- checkpoint and run JSON exist;
- model type is `single_head_causal_attention_lm`;
- source commit and data hashes are recorded;
- final validation loss is finite;
- sample length is exact.

**Step 4: Run to verify failure**

Run:

```powershell
python -m pytest tests/python/test_lesson03_train.py -q
```

Expected: collection fails because `lesson03_train` does not exist.

### Task 5: Implement Lesson 03 training and sampling

**Files:**
- Create: `training/nanogpt_nspire/lesson03_train.py`
- Test: `tests/python/test_lesson03_train.py`

**Step 1: Implement context-aware fixed-seed sampling**

At every step:

1. retain the generated sequence;
2. crop to its last `block_size` tokens;
3. run the whole context;
4. sample from the final-position logits on CPU with a caller-owned generator;
5. append the selected token.

**Step 2: Implement the training run**

Reuse verified batches and shared evaluation. Record:

- exact model component list;
- parameter count and raw FP32 bytes;
- uniform, initial, and final validation loss/BPC;
- fixed-seed sample;
- environment and peak CUDA training allocation;
- checkpoint size/hash;
- dataset hashes and implementation commit.

**Step 3: Keep artifacts ignored**

Write only:

```text
artifacts/lesson03/
├── single_head_attention_lm.pt
└── run.json
```

### Task 6: Write the lesson and commit the implementation

**Files:**
- Create: `docs/lessons/03-causal-self-attention.md`
- Modify: `README.md`

**Step 1: Explain the tensor path**

Explain:

- learned position embeddings;
- query, key, and value;
- scaled dot-product scores;
- the `(T,T)` attention matrix;
- lower-triangular causal masking;
- softmax rows and weighted value sums;
- residual connection;
- why future isolation is a correctness condition.

**Step 2: Explain the exact boundary**

State prominently that the model has one head and no LayerNorm, MLP, dropout, or stacked blocks. It is a causal attention learning model, not yet full nanoGPT.

**Step 3: Verify and commit**

Run:

```powershell
python -m pytest -q
python -m compileall -q training
git diff --check
git add README.md training tests docs
git commit -m "feat: add single-head causal attention model"
```

Expected: a clean implementation commit with no generated artifacts.

### Task 7: Run the real CUDA comparison

**Files:**
- Create: `experiments/lesson03-causal-attention.json`
- Modify: `docs/lessons/03-causal-self-attention.md`

**Step 1: Train from the exact implementation commit**

Initial target configuration:

```powershell
python -m nanogpt_nspire.lesson03_train `
  --data-dir artifacts/data/tinyshakespeare `
  --output-dir artifacts/lesson03 `
  --device auto `
  --seed 1337 `
  --steps 2000 `
  --batch-size 64 `
  --block-size 64 `
  --embedding-dim 64 `
  --learning-rate 0.003 `
  --eval-batches 50 `
  --sample-tokens 300 `
  --source-commit <implementation-commit>
```

**Step 2: Require the contextual comparison**

Use the same dataset and validation-window seed as Lesson 02. Record:

- Lesson 02 final loss/BPC;
- Lesson 03 final loss/BPC;
- parameter and training-cost differences;
- whether context improved validation loss;
- no claim of size-matched deployment superiority.

If the first run is unstable or fails to improve, preserve its `run.json`, diagnose from training history, and change one documented hyperparameter at a time. Do not silently replace a failed run.

**Step 3: Independently verify**

Reload the checkpoint strictly, recompute final loss on the fixed windows, regenerate the sample, and rerun the future-isolation test.

**Step 4: Commit bounded evidence and push**

```powershell
git add experiments/lesson03-causal-attention.json `
  docs/lessons/03-causal-self-attention.md
git commit -m "docs: record lesson 03 causal attention experiment"
git push origin main
python -m pytest -q
git status --short --branch
git ls-remote origin refs/heads/main
```

Expected: all tests pass, generated artifacts remain ignored, and local/remote commits agree.
