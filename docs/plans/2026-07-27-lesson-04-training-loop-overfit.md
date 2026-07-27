# Lesson 04 Training Loop and Overfitting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every stage of one optimization step observable, then deliberately memorize one fixed Tiny Shakespeare batch to distinguish optimization success from generalization.

**Architecture:** Reuse the Lesson 03 single-head causal language model without changing its architecture. Add a small model-agnostic training-loop module that owns forward/loss, backward, optional gradient clipping, optimizer update, parameter-update measurement, fixed-batch evaluation, and repeated fixed-batch fitting. A separate Lesson 04 CLI selects one deterministic batch, measures held-out validation loss, writes a checkpoint and bounded run record, and treats low fixed-batch loss as an overfitting demonstration rather than a good language-model result.

**Tech Stack:** Python 3.10+, PyTorch 2+, pytest 8+, CUDA, JSON, Markdown

---

### Task 1: Specify one observable optimization step

**Files:**
- Create: `tests/python/test_training_loop.py`
- Create: `training/nanogpt_nspire/training_loop.py`

**Step 1: Write the failing gradient and update test**

Use a tiny language model and require one step to report:

```python
metrics = train_step(model, optimizer, inputs, targets, max_grad_norm=1.0)
assert metrics.loss > 0.0
assert metrics.gradient_l2_norm_before_clip > 0.0
assert metrics.gradient_l2_norm_after_clip <= 1.0 + 1e-6
assert metrics.parameter_update_l2_norm > 0.0
assert 0.0 <= metrics.token_accuracy <= 1.0
```

Also compare the reported gradient norm with a manual sum of squared gradients.

**Step 2: Specify mode restoration**

`evaluate_batch` must temporarily enter evaluation mode, disable gradient tracking,
return loss and token accuracy, and restore the caller's original train/eval mode.

**Step 3: Specify failure behavior**

Reject:

- non-positive or non-finite clipping thresholds;
- a model that does not return `(logits, loss)`;
- missing or non-finite loss;
- non-finite gradients before `optimizer.step()`;
- non-positive step and recording intervals.

**Step 4: Run to verify failure**

Run:

```powershell
python -m pytest tests/python/test_training_loop.py -q
```

Expected: collection fails because `nanogpt_nspire.training_loop` does not exist.

### Task 2: Implement the model-agnostic training loop

**Files:**
- Create: `training/nanogpt_nspire/training_loop.py`
- Test: `tests/python/test_training_loop.py`

**Step 1: Add immutable metric records**

Define:

```python
BatchMetrics(loss: float, token_accuracy: float)
TrainStepMetrics(
    loss: float,
    token_accuracy: float,
    gradient_l2_norm_before_clip: float,
    gradient_l2_norm_after_clip: float,
    parameter_update_l2_norm: float,
)
```

**Step 2: Implement the exact step order**

The implementation order must remain visible:

```text
model.train()
optimizer.zero_grad(set_to_none=True)
logits, loss = model(inputs, targets)
validate finite loss
loss.backward()
measure gradients
optional clip_grad_norm_
validate finite gradients
optimizer.step()
measure parameter update
```

Do not add AMP, gradient accumulation, schedulers, distributed training, or callbacks.

**Step 3: Implement fixed-batch fitting**

`overfit_fixed_batch` repeatedly calls `train_step` on exactly the same input and
target tensors. Record step 0, step 1, every `record_every` steps, and the final
step. Return a JSON-friendly history and the final fixed-batch metrics.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests/python/test_training_loop.py -q
```

Expected: all one-step, clipping, mode, failure, and overfitting tests pass.

### Task 3: Specify the Lesson 04 experiment CLI

**Files:**
- Create: `tests/python/test_lesson04_overfit.py`
- Create: `training/nanogpt_nspire/lesson04_overfit.py`

**Step 1: Test configuration validation**

Require positive steps, batch/block/model dimensions, learning rate,
`record_every`, `eval_batches`, and `target_training_loss`. Allow clipping to be
disabled with `None`; otherwise it must be finite and positive.

**Step 2: Add a bounded CPU smoke experiment**

Prepare a tiny character dataset and require:

- a deterministic fixed batch;
- lower final than initial fixed-batch loss;
- checkpoint and `run.json`;
- data hashes, source commit, configuration, model size, and environment;
- initial/final fixed-batch loss and accuracy;
- initial/final held-out validation loss;
- generalization gap and target-loss gate;
- training history with gradient and update norms.

**Step 3: Run to verify failure**

Run:

```powershell
python -m pytest tests/python/test_lesson04_overfit.py -q
```

Expected: collection fails because `lesson04_overfit` does not exist.

### Task 4: Implement the fixed-batch overfitting experiment

**Files:**
- Create: `training/nanogpt_nspire/lesson04_overfit.py`
- Test: `tests/python/test_lesson04_overfit.py`

**Step 1: Select the batch once**

Use `make_batch` exactly once with a caller-owned seeded CPU generator. Reuse the
returned tensors for every optimization step; do not silently resample.

**Step 2: Measure both memorization and generalization**

Before and after fitting, record:

- fixed-batch loss and token accuracy;
- deterministic held-out validation loss/BPC;
- validation minus fixed-batch loss;
- net L2 distance from the initial parameter vector.

The success gate is `final_fixed_batch_loss <= target_training_loss`. Failure to
reach the gate is recorded as `false`; the CLI still preserves artifacts for
diagnosis.

**Step 3: Save reproducible artifacts**

Write:

```text
artifacts/lesson04/
├── overfit_attention_lm.pt
└── run.json
```

Checkpoint fields include model config/state, fixed input/target token IDs,
vocabulary, source commit, and experiment seed.

### Task 5: Teach the complete loop and overfitting

**Files:**
- Create: `docs/lessons/04-training-loop-and-overfitting.md`
- Modify: `README.md`

**Step 1: Explain one update**

Cover:

- why gradients accumulate unless cleared;
- forward pass, scalar cross-entropy, and computation graph;
- reverse-mode autodiff and the chain rule;
- what the optimizer stores and changes;
- gradient norm, clipping, parameter-update norm, and token accuracy;
- why `train()` and `eval()` differ even though this model has no dropout yet.

**Step 2: Explain the overfitting diagnostic**

Clarify that memorizing one fixed batch is a wiring test:

- very low training loss proves the model and optimizer can fit those labels;
- it does not prove held-out generalization;
- a large validation gap is expected and useful here;
- inability to overfit a tiny batch often indicates a bug, insufficient
  capacity, bad optimization settings, or contradictory examples.

**Step 3: Preserve the architecture boundary**

State that Lesson 04 adds no LayerNorm, MLP, multi-head attention, or stacked
blocks. It isolates training mechanics before Lesson 05 quantization.

### Task 6: Commit implementation and run the real experiment

**Files:**
- Create: `experiments/lesson04-overfit.json`
- Modify: `docs/lessons/04-training-loop-and-overfitting.md`

**Step 1: Verify and commit code before training**

Run:

```powershell
python -m pytest -q
python -m compileall -q training tests
python -m nanogpt_nspire.lesson04_overfit --help
git diff --check
git add README.md docs training tests
git commit -m "feat: add observable training loop"
```

Expected: generated artifacts remain ignored and the implementation has a stable
source commit.

**Step 2: Run CUDA overfitting**

Initial target:

```powershell
python -m nanogpt_nspire.lesson04_overfit `
  --data-dir artifacts/data/tinyshakespeare `
  --output-dir artifacts/lesson04 `
  --device auto `
  --seed 1337 `
  --steps 1000 `
  --batch-size 1 `
  --block-size 32 `
  --embedding-dim 64 `
  --learning-rate 0.01 `
  --max-grad-norm 1.0 `
  --record-every 100 `
  --eval-batches 50 `
  --target-training-loss 0.05 `
  --source-commit <implementation-commit>
```

If the target is missed, preserve the failed `run.json`, inspect its history,
and change one documented hyperparameter at a time.

**Step 3: Independently reproduce**

Strictly reload the checkpoint and require:

- stored fixed inputs and targets recompute the recorded final loss/accuracy;
- final validation loss uses the recorded deterministic windows;
- history contains finite gradient and parameter-update norms;
- checkpoint and run hashes match the committed summary;
- artifacts remain ignored.

**Step 4: Record and push**

Commit the bounded JSON record and measured lesson section:

```powershell
git add experiments/lesson04-overfit.json `
  docs/lessons/04-training-loop-and-overfitting.md
git commit -m "docs: record lesson 04 overfitting experiment"
git push origin main
```

Finally rerun the full test suite and verify local, tracking, and remote commits
agree.
