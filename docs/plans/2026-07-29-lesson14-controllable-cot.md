# Lesson 14 Controllable CoT SFT Implementation Plan

> Execute in the `lesson14/controllable-cot` worktree. Write failing tests
> before implementation, preserve the frozen Lesson 12 evaluation file, and
> commit independently verified tasks.

**Goal:** Build and compare direct-answer, short-CoT, and dual-mode SFT students
under equal training and generation-token budgets, then run a bounded
256-to-512 context extension pilot without mixing its result into the primary
CoT comparison.

**Architecture:** Reuse the fixed 264-token byte vocabulary and the reserved
`<THINK>`/`<FINAL>` tokens. Build paired supervised records from exact
project arithmetic/physics and public GSM8K-train rationales. Add a
mode-aware evaluator that treats the mode token as an input prefix and scores
only the final segment. Extend the learned position table in a separate route
and calculate the exact incremental-runtime memory delta.

**Tech stack:** Python 3.12, PyTorch 2.11/CUDA 12.8, NumPy, pytest, existing
packed-corpus/stage-training infrastructure, and the existing Host C tests.

## Step 1: Freeze the mode serialization contract

- Output: mode-aware serializer and tests for token order, assistant-only loss,
  context limits, and `<FINAL>` transition targets.
- Test: direct and reasoning fixtures have exact token/mask arrays; invalid
  modes and over-context records fail.

## Step 2: Build deterministic Lesson 14 corpora

- Output: direct, CoT, hybrid 256-token corpora plus a hybrid 512-token corpus
  and provenance manifest.
- Test: byte-identical rebuild, family-level split isolation, exact answer
  verification, GSM8K-test exclusion, evaluation-family exclusion, and
  context rejection counts.

## Step 3: Implement mode-aware frozen evaluation

- Output: evaluator for `<FINAL>`-prefixed direct output and
  `<THINK>...<FINAL>...` output.
- Test: final-only scoring, missing-transition rejection, truncation
  classification, role/special-token leak handling, and deterministic selected
  prompts.

## Step 4: Train the three 256-token SFT routes

- Output: Direct-Control-SFT, Short-CoT-SFT, and Hybrid-Control-SFT
  checkpoints from the exact Lesson 12 CPT parent.
- Test: strict checkpoint lineage/config reload, fixed 1,000 updates and
  4.096M sampled tokens, finite losses, and full validation/test summaries.

## Step 5: Evaluate equal-token CoT benefit and controllability

- Output: one comparison artifact with 48-token primary and 96-token
  diagnostic results.
- Test: 128 identical families per evaluation, direct/CoT mode compliance,
  exact per-task counts, truncation counts, and deterministic JSON hashes.

## Step 6: Run the 512-token context pilot

- Output: deterministic extended CPT initialization, 512-token CPT
  continuation, optional hybrid SFT checkpoint, and exact parameter/KV-arena
  estimates.
- Test: all non-position tensors and rows 0--255 are byte-identical to the
  parent, new rows are finite/reproducible, and the pilot never enters the
  primary CoT score table.

## Step 7: Teach, verify, and publish

- Output: `docs/lessons/14-controllable-cot-sft.md`, experiment JSON, README
  update, clean branch merged to main and pushed.
- Test: full Python suite, Host C CTest suite, secret scan, artifact/hash
  cross-check, clean Git status, and `origin/main == main`.

