# Lesson 07 Teacher v2 and Conditional Distillation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run one preregistered dropout-only Teacher v2 experiment and, only if it passes the unchanged source-quality gate, train a same-architecture Distilled-Small student.

**Architecture:** Reuse the shared GPT trainer for a 6×384 Teacher v2 whose only experimental change from v1 is dropout `0.2 -> 0.3`. Preserve a hard stop between Teacher selection and distillation. If unlocked, reuse the exact Direct-Small student architecture and training protocol while replacing its hard-only objective with a frozen Teacher-guided hard/soft loss.

**Tech Stack:** Python 3.10+, PyTorch 2+, pytest 8+, CUDA, JSON, Markdown

---

### Task 1: Freeze and test Teacher v2

**Files:**
- Create: `training/nanogpt_nspire/teacher_v2_train.py`
- Create: `tests/python/test_teacher_v2_train.py`

**Step 1: Write failing profile tests**

Require:

```text
route                 = Teacher-v2
checkpoint            = teacher_v2_gpt.pt
dropout               = 0.3
quality threshold     = 1.4797899746894836
parameters            = 10,695,936
training tokens       = 81,920,000
```

Compare v1 and v2 dataclass fields and require only `dropout` and
`output_dir` to differ. Construct both models from seed 1337 and require every
initial Parameter to be exactly equal.

**Step 2: Implement the frozen entry**

Build v2 with `dataclasses.replace(frozen_teacher_config(...))`. Do not copy
the base training loop or expose architecture/hyperparameter CLI overrides.

**Step 3: Add a CPU smoke run**

Use a deliberately small test config with the Teacher-v2 identity. Require
the separate checkpoint name, route, source commit, threshold, pass/fail
field and strict reload.

**Step 4: Run focused and full tests**

Run:

```powershell
python -m pytest tests/python/test_teacher_v2_train.py -q
python -m pytest -q
python -m compileall -q training tests/python
python -m nanogpt_nspire.teacher_v2_train --help
git diff --check
```

**Step 5: Commit**

```text
feat: add dropout-only teacher v2 profile
```

### Task 2: Train and independently verify Teacher v2

**Files:**
- Create at runtime: `artifacts/lesson07-teacher-v2/teacher_v2_gpt.pt`
- Create at runtime: `artifacts/lesson07-teacher-v2/run.json`

**Step 1: Train from the implementation commit**

Use the exact implementation commit as `source_commit`.

**Step 2: Evaluate the unchanged gate**

Require:

```text
selected validation loss <= 1.4797899746894836
```

Do not reinterpret or lower it.

**Step 3: Independently reproduce**

Strictly reload and reproduce parameter count, tying, loss, sample, causal
future isolation, artifact bytes/hash and source commit.

**Step 4: Apply the hard branch**

- fail: skip Tasks 3–5 and document the failure;
- pass: continue to Task 3 without changing the frozen distillation design.

### Task 3: Implement the distillation objective conditionally

**Files:**
- Create: `training/nanogpt_nspire/distillation.py`
- Create: `tests/python/test_distillation.py`

**Step 1: Test temperature-scaled KL**

Require finite scalar hard, soft and combined losses; reject incompatible
logit/target shapes, nonpositive temperature and alpha outside `[0,1]`.
Verify the soft term approaches zero for equal logits and backpropagates only
through student logits.

**Step 2: Implement**

Use:

```text
temperature = 2.0
alpha       = 0.5
```

Compute Teacher probabilities under no-grad and use `batchmean` KL with
`T^2` scaling.

### Task 4: Add and train Distilled-Small conditionally

**Files:**
- Create: `training/nanogpt_nspire/distilled_small_train.py`
- Create: `tests/python/test_distilled_small_train.py`
- Create at runtime: `artifacts/lesson07-distilled-small/distilled_small_gpt.pt`
- Create at runtime: `artifacts/lesson07-distilled-small/run.json`

**Step 1: Validate the Teacher source**

Require Teacher-v2 route, frozen architecture, passed unchanged quality gate,
strict state load, vocabulary/dataset hashes and source artifact SHA.

**Step 2: Freeze the student**

Require the exact Direct-Small model/training configuration, initialization,
batch sequence, optimizer, schedule, 5,000 steps and 40,960,000 student
tokens.

**Step 3: Train and report both objectives**

Record total, hard and soft training loss at every log point. Teacher remains
in eval mode with no gradients or optimizer state.

**Step 4: Evaluate**

Require deterministic best-checkpoint selection, fixed-seed sample and a
measured comparison to Direct-Small. Do not claim success merely because the
combined training objective fell.

### Task 5: Record Lesson 07 and push

**Files:**
- Create: `docs/lessons/07-teacher-v2-and-distillation.md`
- Create: `experiments/lesson07-teacher-v2.json`
- Conditionally create: `experiments/lesson07-distilled-small.json`
- Modify: `experiments/small-model-comparison.json`
- Modify: `README.md`

**Step 1: Preserve the branch outcome**

Document v2 whether it passes or fails. If it fails, explicitly state that
distillation was not run.

**Step 2: Cross-check evidence**

Match committed metrics, hashes, bytes, source commits and route status
against ignored runtime artifacts.

**Step 3: Final validation**

Run the full tests, compile, check ignored artifacts and require clean diff
checks.

**Step 4: Commit and push**

Push only after local, tracking and remote `main` hashes are verified equal.
