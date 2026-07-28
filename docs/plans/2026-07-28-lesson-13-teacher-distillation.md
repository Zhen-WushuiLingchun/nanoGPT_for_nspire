# Lesson 13 External and Local Teacher Distillation Implementation Plan

> **Execution:** Use the `executing-plans` and `Code` skills. Work in the
> dedicated `lesson13/teacher-distillation` worktree, write tests before each
> implementation, keep API credentials out of files and command arguments,
> and commit each independently verified task.

**Goal:** Compare two genuinely different ways to teach the same deployable
10.8M-parameter student: verified answer sequences from a stronger external
teacher and token-level probability targets from a larger local teacher that
shares the student's 264-token vocabulary.

**Architecture:** Keep the Lesson 12 student architecture, tokenizer, frozen
evaluation suite, CPT parent checkpoint, update count, and sampled-token budget
fixed. Build a credential-isolated DeepSeek V4-Pro sequence-teacher pipeline
whose answers must pass deterministic math/physics checks before entering SFT.
Separately train a larger local GPT with the exact project tokenizer, then
distill its assistant-token logits into a fresh student initialized from the
same CPT checkpoint. Treat a combined sequence-plus-logit run as a fourth
ablation, not as evidence for either component alone.

**Tech stack:** Python 3.12, PyTorch CUDA 12.8, NumPy, standard-library HTTPS,
DeepSeek OpenAI-compatible Chat Completions, exact `Decimal`/unit verifiers,
pytest, and the existing `DirectSmallGPT`, packed corpus, stage trainer, and
assistant evaluator.

**Claim boundary:** External teacher output is sequence-level supervision, not
logit distillation, because the provider tokenizer and probability vector are
not available to the student. Local KL training is genuine logit distillation.
Neither route by itself proves broad reasoning, calculator performance, or
RL-generated chain-of-thought.

---

## Task 1: Freeze security, provider, and experiment contracts

**Files:**

- Modify: `.gitignore`
- Create: `training/nanogpt_nspire/secret_safety.py`
- Create: `tests/python/test_secret_safety.py`
- Create: `experiments/lesson13-teacher-distillation.json`

**Steps:**

1. Ignore `.env`, environment variants, credentials, API caches, raw provider
   responses, generated teacher corpora, checkpoints, and logs.
2. Read `DEEPSEEK_API_KEY` only at request time from process or Windows user
   environment. Never accept it in serializable configuration, CLI arguments,
   checkpoints, manifests, exceptions, or log messages.
3. Add recursive credential-shape scanning and sanitized provider errors.
4. Freeze the current official provider contract: base URL
   `https://api.deepseek.com`, model `deepseek-v4-pro`, explicit thinking mode,
   bounded reasoning effort, non-streaming Chat Completions, and JSON output.
5. Record a maximum request count and token/cost estimate before any paid call.

**Test:** Synthetic secrets are rejected from nested mappings and generated
files; serialized safe configurations contain no secret value, authorization
header, environment-variable assignment, or credential-shaped token.

**Commit:**

```text
security: isolate Lesson 13 teacher credentials
```

## Task 2: Build a verified external sequence-teacher corpus

**Files:**

- Create: `training/nanogpt_nspire/external_teacher.py`
- Create: `training/nanogpt_nspire/lesson13_sequence_data.py`
- Create: `tests/python/test_external_teacher.py`
- Create: `tests/python/test_lesson13_sequence_data.py`

**Steps:**

1. Select training-only arithmetic and physics families; fail if any frozen
   Lesson 12 evaluation family appears.
2. Ask V4-Pro for concise English answers with a final machine-checkable
   answer. Do not ask it to invent ground truth that the project can calculate.
3. Parse strict JSON, preserve provider model/request/token provenance, and
   discard private reasoning content from the training record.
4. Accept math only when the final numeric answer matches exact project ground
   truth. Accept physics only when the numeric answer and canonical unit match.
5. Quarantine malformed, disagreeing, duplicate, over-context, or role-leaking
   responses. Build deterministic `<USER>`/`<ASSISTANT>` packed shards from
   accepted outputs.
6. Support a no-key dry run that writes only request plans and never performs
   network calls.

**Test:** A fake transport covers valid, invalid, retryable, unauthorized,
malformed, duplicate, leaked-role, and evaluation-leak cases without using a
real credential.

**Commit:**

```text
data: build verified external teacher sequences
```

## Task 3: Train a shared-tokenizer local teacher

**Files:**

- Create: `training/nanogpt_nspire/local_teacher_train.py`
- Create: `tests/python/test_local_teacher_train.py`

**Steps:**

1. Freeze a larger 12-layer, 10-head, 640-wide local GPT with the same
   264-token vocabulary and 256-token context as the student.
2. Train it on the exact Lesson 12 CPT mix, then role-aware SFT, with explicit
   parent hashes and route names. Do not reuse Tiny Shakespeare teachers.
3. Select checkpoints only by the same full validation loss contract and
   record throughput, VRAM, wall time, parameter bytes, and lineage.
4. Verify tied embeddings, architecture, tokenizer identity, causal isolation,
   exact checkpoint reload, and assistant-generation formatting.

**Test:** CPU smoke training and incompatible-parent tests pass before the
bounded CUDA run.

**Commit:**

```text
feat: train shared-tokenizer local teacher
```

## Task 4: Compare hard, sequence, logit, and combined students

**Files:**

- Modify: `training/nanogpt_nspire/distillation.py`
- Create: `training/nanogpt_nspire/lesson13_distill_train.py`
- Create: `tests/python/test_lesson13_distill_train.py`

**Steps:**

1. Initialize every student from the exact Lesson 12 CPT checkpoint.
2. Keep architecture, optimizer, update count, batch shape, sampled tokens,
   validation selection, and evaluation prompts fixed.
3. Compare:
   - ordinary hard-label SFT (Lesson 12 baseline);
   - external verified sequence SFT;
   - original SFT with local teacher KL on assistant targets;
   - combined external sequence plus local teacher KL.
4. Apply both hard cross-entropy and temperature-scaled KL only where the
   assistant loss mask is one; the local teacher remains frozen in eval mode.
5. Record hard loss, KL loss, teacher forward cost, student wall time, and
   exact lineage. Do not claim that an external sequence run used teacher
   logits.

**Test:** Masked user positions have zero hard and KL contribution, teacher
parameters receive no gradients, and alpha zero reproduces hard-label loss.

**Commit:**

```text
feat: compare sequence and logit distillation
```

## Task 5: Run the bounded experiment and teach the result

**Files:**

- Create: `docs/lessons/13-external-and-local-teachers.md`
- Modify: `README.md`

**Steps:**

1. Run the external-teacher request plan and require explicit credential
   presence before the paid generation step.
2. Train the local teacher and the three new student routes on CUDA.
3. Evaluate all checkpoints on the immutable Lesson 12 suite with the same
   greedy token limit.
4. Explain teacher versus student, sequence versus logit distillation,
   tokenizer compatibility, temperature/KL, verification and rejection, and
   why a stronger teacher can still teach bad shortcuts.
5. Run all Python and Host C tests, scan tracked/generated metadata for
   credential-shaped values, commit, merge to `main`, and push.

**Test:** All gates pass; checkpoint and evaluation hashes match the committed
experiment record; Git is clean and `origin/main` equals local `main`.

**Commit:**

```text
docs: teach external and local distillation
```

## Frozen downstream order

1. **Lesson 14:** direct-answer versus CoT SFT/RLVR under equal generated-token
   limits.
2. **Lesson 15:** real W4A8 export, PyTorch/Host C alignment, safe arithmetic
   tool comparison, and physical Nspire deployment.
