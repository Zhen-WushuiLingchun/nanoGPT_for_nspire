# Lesson 15 implementation plan

## Step 1: Freeze long-output evidence

- Output: 256-token Direct/Think results and 384-token Think diagnostic for the
  Lesson 14 512 checkpoint.
- Test: all 128 prompts are identical to the frozen evaluation hash; primary
  context truncation is zero.

## Step 2: Implement the efficient attention model

- Output: `EfficientLongContextGPT` with 6 query heads, configurable 1/2/6 K/V
  heads, and learned/ALiBi position modes.
- Test: causal isolation, tensor shapes, deterministic ALiBi slopes, parameter
  formula, full-MHA equivalence when `n_kv_head == n_head`.

## Step 3: Convert the MHA checkpoint

- Output: deterministic GQA-Learned and GQA-ALiBi init checkpoints.
- Test: exact copy of all compatible tensors; K/V group means equal the source
  head averages; hash/route/config validation rejects drift.

## Step 4: Add frozen training and evaluation

- Output: 250-step CPT and 1,000-step Hybrid SFT routes for both variants,
  position-bucket loss evaluation, and 256-token reasoning evaluation.
- Test: identical data, seed, optimizer settings, sampled token count, and
  evaluation families.

## Step 5: Run and compare

- Output: ignored checkpoints/logs plus a tracked machine-readable experiment
  summary.
- Test: process completion, artifact hashes, loss metrics, exact evaluation,
  and deterministic conversion rebuild.

## Step 6: Teach and verify

- Output: Lesson 15 Markdown, README update, and accepted/updated ADR.
- Test: Python suite, Host C CTest, JSON validation, Markdown links, secret
  scan, clean Git state, merge, and remote push.
