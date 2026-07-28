# Lesson 11 English Base Pilot Implementation Plan

> **Execution:** Use the `executing-plans` skill and complete each task with
> focused tests, compact evidence, and a separate commit.

**Goal:** Prove that the frozen 264-token English data contract can train a
real causal base language model on the RTX 5080 Laptop GPU, while measuring the
deployable student's model-file and inference-RAM budget before expensive
training begins.

**Architecture:** Reuse `DirectSmallGPT` as the causal decoder engine. Add a
pure budget estimator, a deterministic public-corpus sampler that produces the
Lesson 10 shard format, and an English base-training command that reads those
shards with memory mapping. The committed pilot is deliberately bounded:
FineWeb-Edu supplies general educational prose and OpenWebMath supplies
math-heavy prose; raw documents and checkpoints remain ignored artifacts.

**Tech stack:** Python 3.12, PyTorch CUDA 12.8 wheels, NumPy, Hugging Face Hub,
PyArrow/Parquet, pytest, the existing byte tokenizer and corpus manifest.

**Claim boundary:** A lower validation loss proves that next-byte pretraining
worked on the sampled corpus. It does not yet prove instruction following,
arithmetic correctness, physics understanding, distillation benefit, or
calculator deployment.

---

## Task 1: Isolate and verify the CUDA training environment

**Files:**

- Create: `training/nanogpt_nspire/cuda_probe.py`
- Create: `tests/python/test_cuda_probe.py`
- Create: `experiments/lesson11-cuda-environment.json`

**Steps:**

1. Write tests for a bounded, JSON-serializable environment report and for
   failure when CUDA is required but unavailable.
2. Create an ignored worktree-local `.venv`.
3. Install the official PyTorch CUDA 12.8 wheel and only the data dependencies
   required by this lesson. Do not modify the main repository environment.
4. Run a deterministic matrix multiplication, synchronize, and record device,
   compute capability, runtime CUDA version, PyTorch version, peak allocated
   memory, elapsed time, and a finite checksum.
5. Commit only the probe, tests, and compact evidence; never commit the virtual
   environment or package cache.

**Commit:**

```text
test: verify Lesson 11 CUDA environment
```

## Task 2: Freeze model, file, and inference-memory budgets

**Files:**

- Create: `training/nanogpt_nspire/model_budget.py`
- Create: `tests/python/test_model_budget.py`
- Create: `experiments/lesson11-model-budget.json`

**Steps:**

1. Test exact parameter counts against instantiated `DirectSmallGPT` models,
   including tied versus untied embeddings.
2. Estimate FP32/FP16/INT8/W4 groupwise storage per tensor. Count W4 scales,
   tensor-table/alignment overhead, embeddings/norms kept at higher precision,
   KV cache, residual workspaces, logits, and a named safety reserve.
3. Reject invalid head divisibility, unsupported storage policies, nonpositive
   dimensions, and estimates that omit a required memory component.
4. Compare a small architecture grid and freeze:
   - deployable student candidate: 6 layers, 6 heads, width 384, context 256;
   - computer-only teacher candidate: 12 layers, 10 heads, width 640,
     context 256.
5. Record whether the student satisfies the 4--6 MiB file band and the chosen
   calculator RAM ceiling. Keep the result labelled an estimate until `.ngm`
   v2 is exported and measured in Lesson 15.

**Commit:**

```text
feat: freeze English model deployment budgets
```

## Task 3: Ingest a bounded, reproducible public pilot corpus

**Files:**

- Create: `training/nanogpt_nspire/public_corpus.py`
- Create: `training/nanogpt_nspire/lesson11_data.py`
- Create: `tests/python/test_public_corpus.py`
- Create: `tests/python/test_lesson11_data.py`
- Create: `experiments/lesson11-public-pilot-data.json`

**Steps:**

1. Test document normalization, quality filters, stable IDs, exact source
   revisions, deterministic selection, document-boundary serialization, and
   train/validation/test family isolation using synthetic rows.
2. Fetch only pinned Parquet files from official dataset repositories:
   FineWeb-Edu `sample-10BT` with high educational score and OpenWebMath.
3. Keep source URL, revision, source document ID/URL, license, and selection
   rule in every record. Fail closed when a required provenance field is
   absent.
4. Build a small smoke corpus first, then a bounded real pilot corpus in
   ignored `artifacts/lesson11-public-pilot/`. Do not download either full
   dataset.
5. Rebuild from the cached, pinned inputs and require identical manifest and
   shard hashes. Commit counts, bytes, source mixture, hashes, and exclusions,
   but no raw documents or binary shards.

**Commit:**

```text
data: build pinned English public corpus pilot
```

## Task 4: Train and evaluate the first English Base checkpoint

**Files:**

- Create: `training/nanogpt_nspire/base_train.py`
- Create: `tests/python/test_base_train.py`
- Create: `experiments/lesson11-base-pilot.json`

**Steps:**

1. Test little-endian `uint16` memory mapping, document-safe batch sampling,
   deterministic seeding, byte-token vocabulary checks, resume metadata, and
   CPU smoke training.
2. Add CUDA AMP, gradient accumulation, clipping, cosine decay, periodic
   validation, best-checkpoint selection, and atomic run metadata.
3. Run a tiny CUDA overfit gate. Require loss to fall on one repeated batch
   before spending time on the public corpus.
4. Run a bounded base pilot with the frozen student architecture. Record
   initial/best/final validation cross-entropy, byte perplexity,
   bits-per-byte, tokens processed, tokens/s, wall time, peak VRAM, and fixed
   prompt continuations.
5. Compare against the uniform 264-token baseline and a deterministic
   frequency baseline. Keep all checkpoints under ignored `artifacts/`.

**Commit:**

```text
feat: train first English base model pilot
```

## Task 5: Teach the result and run regression gates

**Files:**

- Create: `docs/lessons/11-english-base-pilot.md`
- Modify: `README.md`

**Steps:**

1. Explain what random initialization, causal next-byte loss, an optimizer
   step, a token budget, an epoch, validation loss, byte perplexity, and
   bits-per-byte mean.
2. Explain why the pilot is a genuine base language model but is not yet a
   chatbot, why role-aware SFT comes next, and why a low loss alone does not
   establish arithmetic or physics accuracy.
3. Explain Student versus Teacher capacity and why W4 file size differs from
   FP32 training memory and calculator inference RAM.
4. Record all commands, pinned revisions, hashes, GPU observations, failed
   gates, and honest limitations.
5. Run all Python tests, CMake/CTest runtime regressions, `git diff --check`,
   credential scanning, and repository hygiene checks.

**Commit:**

```text
docs: teach English base pretraining pilot
```

## Deferred after Lesson 11

1. **Lesson 12:** continued math/physics pretraining plus role-aware SFT.
2. **Lesson 13:** verified external sequence teacher data and local
   shared-tokenizer teacher/logit distillation.
3. **Lesson 14:** direct versus CoT SFT/RLVR under equal generated-token
   budgets.
4. **Lesson 15:** byte/special-token `.ngm` v2, measured W4A8 export, safe C
   calculator tool, Host C alignment, and physical Nspire deployment.
