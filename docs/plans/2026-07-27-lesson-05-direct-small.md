# Lesson 05 Direct-Small Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and train the first complete, budgeted Direct-Small GPT that later serves as the identical student architecture for distillation.

**Architecture:** Implement a nanoGPT-style pre-norm decoder with fused multi-head QKV attention, tanh GELU MLPs, residual connections, final LayerNorm, and tied token/output weights. Freeze the deployable v1 configuration at 4 layers, 5 heads, width 160, context 128, bias-free parameters, and dropout 0.1. Train from random parameters with a deterministic AdamW/cosine protocol, select the best fixed-window validation checkpoint, and record only preliminary deployment eligibility until the C model file and runtime exist.

**Tech Stack:** Python 3.10+, PyTorch 2+, pytest 8+, CUDA, JSON, Markdown

---

### Task 1: Specify the complete small GPT

**Files:**
- Create: `tests/python/test_direct_small_gpt.py`
- Create: `training/nanogpt_nspire/models/direct_small_gpt.py`
- Modify: `training/nanogpt_nspire/models/__init__.py`

**Step 1: Test configuration validation**

Reject:

- non-positive vocabulary, context, layer, head, width, or MLP ratio;
- `n_embd` not divisible by `n_head`;
- dropout outside `[0,1)`;
- non-boolean bias or weight tying.

**Step 2: Test multi-head shapes and causality**

For `(B,T,C)`, require:

```text
q,k,v:   (B,H,T,C/H)
weights: (B,H,T,T)
output:  (B,T,C)
```

Every weights row sums to one and the strict upper triangle is zero. Changing
future tokens must not change earlier logits while the model is in eval mode.

**Step 3: Test the block and parameter formula**

For bias-free tied weights:

```text
P = V*C + T*C + L*(12*C² + 2*C) + C
```

The frozen v1 config must produce exactly:

```text
parameters = 1,261,120
raw FP32 bytes = 5,044,480
```

Require `token_embedding.weight` and `lm_head.weight` to be the same parameter,
not merely equal copies.

**Step 4: Test full forward/backward**

Require `(B,T,V)` logits, finite scalar cross-entropy, populated finite
gradients, invalid-input rejection, and a tiny contextual task whose loss falls
substantially.

**Step 5: Run to verify failure**

Run:

```powershell
python -m pytest tests/python/test_direct_small_gpt.py -q
```

Expected: collection fails because `direct_small_gpt` does not exist.

### Task 2: Implement the Direct-Small reference model

**Files:**
- Create: `training/nanogpt_nspire/models/direct_small_gpt.py`
- Modify: `training/nanogpt_nspire/models/__init__.py`
- Test: `tests/python/test_direct_small_gpt.py`

**Step 1: Add explicit components**

Implement:

- optional-bias LayerNorm via `torch.nn.functional.layer_norm`;
- fused bias-free `C -> 3C` QKV projection;
- manual scaled multi-head causal attention;
- `C -> 4C -> C` MLP with `GELU(approximate="tanh")`;
- pre-norm Transformer block;
- token/position embeddings, four blocks, final norm, and vocabulary head.

Keep the causal mask non-persistent because it is derived from `block_size`.

**Step 2: Tie and initialize weights**

Tie the token embedding and vocabulary head parameter. Initialize Linear and
Embedding weights from `Normal(0,0.02)`, zero enabled biases, and apply the
nanoGPT residual-projection scale `0.02 / sqrt(2*n_layer)` to attention and MLP
output projections.

**Step 3: Preserve the reference path**

Do not use Flash Attention, fused LayerNorm, AMP, quantization, or compile in the
reference implementation. Optimized variants may be added later only alongside
numeric alignment tests.

### Task 3: Specify optimizer and schedule behavior

**Files:**
- Create: `tests/python/test_direct_small_train.py`
- Create: `training/nanogpt_nspire/direct_small_train.py`

**Step 1: Test cosine warmup**

Require:

```text
step 1              -> max_lr / warmup_steps
step warmup_steps   -> max_lr
step max_steps      -> min_lr
after max_steps     -> min_lr
```

Reject inconsistent or non-finite schedule arguments.

**Step 2: Test AdamW grouping**

Matrix/embedding weights receive weight decay; LayerNorm and Linear bias vectors
do not. Every unique trainable parameter appears exactly once, including tied
weights.

**Step 3: Test training configuration**

Validate the full architecture, update/evaluation intervals, optimizer values,
sample settings, 6 MiB file cap, and 64 KiB metadata reserve before loading data.

### Task 4: Implement reproducible Direct-Small training

**Files:**
- Create: `training/nanogpt_nspire/direct_small_train.py`
- Test: `tests/python/test_direct_small_train.py`

**Step 1: Add a bounded CPU smoke run**

Use a tiny configuration and require:

- deterministic train/evaluation generators;
- periodic training and validation history;
- best validation step and selected checkpoint;
- fixed-seed context sample;
- exact parameter/raw-byte calculation;
- deployment estimate and eligibility;
- checkpoint and `run.json`.

**Step 2: Implement the efficient update loop**

Each update remains explicit:

```text
set scheduled learning rate
sample batch
zero_grad(set_to_none=True)
forward
finite-loss check
backward
clip global gradient norm
optimizer.step
```

Do not copy all parameters every step as Lesson 04 did for observability.

**Step 3: Select the best validation checkpoint**

Evaluate the same seeded validation windows at step 0, every `eval_interval`,
and the final step. Preserve a CPU copy only when loss improves. After training,
load the selected weights and recompute validation loss before sampling and
saving.

**Step 4: Record evidence boundaries**

Record:

- final-step and selected validation loss/BPC;
- training tokens, update time, evaluation time, and peak CUDA allocation;
- dataset hashes and source commit;
- raw FP32 bytes and `raw + 64 KiB` deployment estimate;
- `Host measured peak RAM`, `Nspire measured peak RAM`, C alignment, and actual
  deployment file as explicit pending fields.

### Task 5: Teach the complete Direct-Small architecture

**Files:**
- Create: `docs/lessons/05-direct-small-gpt.md`
- Create: `experiments/small-model-comparison.json`
- Modify: `README.md`

**Step 1: Explain every new model component**

Cover:

- splitting fused Q/K/V into five heads;
- head dimension 32 and concatenation;
- pre-norm residual order;
- MLP expansion and tanh GELU;
- final LayerNorm;
- tied embedding/output weights;
- dropout during training and disabled inference;
- parameter formula and preliminary weight/KV budget.

**Step 2: Add the evolving comparison table**

Create a machine-readable comparison record with:

- the fixed two-layer fairness protocol;
- Direct-Small frozen architecture;
- Quantized-Small and Distilled-Small status `pending`;
- teacher candidate status `provisional`;
- Host/C/Nspire fields carrying reasoned pending states rather than zeros.

**Step 3: Update course numbering**

Mark Lesson 05 as Direct-Small and shift quantization, distillation, C alignment,
and Nspire measurement to Lessons 06–09.

### Task 6: Verify and commit implementation

Run:

```powershell
python -m pytest -q
python -m compileall -q training tests
python -m nanogpt_nspire.direct_small_train --help
git diff --check
git add README.md docs experiments training tests
git commit -m "feat: add deployable direct small GPT"
```

Expected: generated artifacts remain ignored and the implementation source commit
is stable before the real run.

### Task 7: Run and verify the CUDA baseline

**Files:**
- Create: `experiments/lesson05-direct-small.json`
- Modify: `experiments/small-model-comparison.json`
- Modify: `docs/lessons/05-direct-small-gpt.md`

**Step 1: Train from the implementation commit**

Run:

```powershell
python -m nanogpt_nspire.direct_small_train `
  --data-dir artifacts/data/tinyshakespeare `
  --output-dir artifacts/lesson05-direct-small `
  --device auto `
  --seed 1337 `
  --steps 5000 `
  --batch-size 64 `
  --block-size 128 `
  --n-layer 4 `
  --n-head 5 `
  --n-embd 160 `
  --dropout 0.1 `
  --learning-rate 0.001 `
  --min-learning-rate 0.0001 `
  --warmup-steps 100 `
  --weight-decay 0.1 `
  --beta1 0.9 `
  --beta2 0.99 `
  --max-grad-norm 1.0 `
  --eval-interval 250 `
  --eval-batches 50 `
  --log-interval 100 `
  --sample-tokens 300 `
  --temperature 0.8 `
  --source-commit <implementation-commit>
```

This processes exactly `40,960,000` training tokens.

**Step 2: Preserve failures**

If loss becomes non-finite, the size gate fails, or selected validation loss does
not improve materially over the Lesson 03 model, preserve `run.json` and diagnose
one variable at a time. Do not silently replace the protocol.

**Step 3: Independently reproduce**

Strictly reload the selected checkpoint and require:

- model config and unique parameter count match;
- tied weights remain tied after load;
- selected validation loss exactly reproduces on fixed windows;
- fixed-seed sample exactly reproduces;
- future-isolation still passes;
- checkpoint/run bytes and SHA-256 match the committed record;
- artifacts remain ignored.

**Step 4: Commit and push**

Run the full suite again, commit the bounded experiment evidence, push `main`,
and verify local, tracking, and remote commits agree.
