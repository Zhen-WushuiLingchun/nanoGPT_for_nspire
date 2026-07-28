# Lesson 12 Math/Physics CPT and Role-Aware SFT Implementation Plan

> **Execution:** Use the `executing-plans` and `Code` skills. Work in the
> dedicated `lesson12/math-physics-sft` worktree, write tests before each
> implementation, keep downloaded data and checkpoints under ignored
> `artifacts/`, and commit each independently verified task.

**Goal:** Turn the Lesson 11 English next-byte model into the first narrow
English assistant baseline by separating two attributable interventions:
continued pretraining (CPT) on math/physics material and supervised fine-tuning
(SFT) with real `<USER>` / `<ASSISTANT>` tokens and assistant-only loss.

**Architecture:** Keep the frozen deployable student architecture and
264-token byte vocabulary unchanged. Build a replay-aware CPT corpus from the
exact Lesson 11 general-English shards plus bounded domain records. Build a
separate role-aware SFT corpus from exact arithmetic/physics templates and
permissively licensed public examples. Add a stage trainer that loads a
declared parent checkpoint with strict architecture/tokenizer/hash checks, then
evaluate the Base, CPT, and SFT checkpoints on the same frozen prompt suite.

**Tech stack:** Python 3.12, PyTorch CUDA 12.8, NumPy, PyArrow, Hugging Face
Hub, exact `Decimal`/`Fraction` verifiers, pytest, and the existing
`DirectSmallGPT`, tokenizer, corpus, and Lesson 11 trainer.

**Claim boundary:** This lesson may show that domain continuation changes
math/physics byte loss and that role-aware SFT improves narrow frozen tasks. It
does not establish broad reasoning, teacher-distillation benefit, RLVR/CoT
benefit, W4A8 alignment, or calculator-side performance.

---

## Task 1: Freeze source snapshots and the Lesson 12 evaluation contract

**Files:**

- Create: `training/nanogpt_nspire/lesson12_curriculum.py`
- Create: `tests/python/test_lesson12_curriculum.py`
- Create: `experiments/lesson12-data.json`

**Steps:**

1. Test strict parsers for project arithmetic/physics records, GSM8K final
   answers, and short English OASST1 prompt/assistant pairs.
2. Pin every public file to an exact repository commit and record file hashes.
   Keep GSM8K's original test split outside all training shards.
3. Generate deterministic arithmetic and introductory-physics families with
   exact answers, units, formula metadata, and family-level split isolation.
4. Freeze an evaluation JSONL containing held-out arithmetic, physics, role
   boundary, and short-conversation prompts. Store only its schema, counts,
   family hashes, and file hash in Git; the full examples remain ignored.
5. Fail closed on malformed UTF-8, missing provenance, unsafe arithmetic
   parsing, over-context examples, duplicate families, or source/test leakage.

**Test:** Two independent builds from the same cached pinned inputs have
byte-identical manifests, shards, evaluation JSONL, and hashes.

**Commit:**

```text
data: freeze Lesson 12 math physics curriculum
```

## Task 2: Build distinct replay-aware CPT and role-aware SFT corpora

**Files:**

- Create: `training/nanogpt_nspire/lesson12_data.py`
- Create: `tests/python/test_lesson12_data.py`

**Steps:**

1. Add a verified shard-composition path that combines the exact Lesson 11
   corpus with domain base records without decoding or rewriting the source
   text.
2. Record component manifest hashes and token contributions. Target a
   meaningful general-English replay fraction and report the measured value;
   do not infer it from record counts.
3. Serialize SFT examples as
   `<BOS><USER>question<ASSISTANT>answer<EOS>`.
4. Require that only assistant bytes and the final `<EOS>` contribute to SFT
   loss, and report total versus eligible target counts per split.
5. Preserve family IDs across question paraphrases and between CPT/SFT/eval so
   related variants cannot silently cross frozen boundaries.

**Test:** Every shard is little-endian, hash-stable, role masks are exact, all
three splits are nonempty, and the frozen public test split is absent.

**Commit:**

```text
feat: build replay CPT and role-aware SFT shards
```

## Task 3: Add strict parent-checkpoint stage training

**Files:**

- Create: `training/nanogpt_nspire/stage_train.py`
- Create: `tests/python/test_stage_train.py`
- Modify: `pyproject.toml`

**Steps:**

1. Test loading a parent checkpoint only when its model configuration,
   vocabulary contract, tensor keys/shapes, source file hash, and declared
   route match.
2. Initialize CPT from the Lesson 11 Base checkpoint and SFT from the selected
   CPT checkpoint. Never reinitialize model weights between stages.
3. Reuse masked cross-entropy so CPT trains on domain/replay bytes while SFT
   trains only on assistant targets.
4. Use stage-specific learning rates, validation selection, gradient clipping,
   CUDA bfloat16, atomic checkpoints, and deterministic run metadata.
5. Save the parent hash, data-manifest hash, training tokens, eligible-answer
   tokens, best step, full validation/test loss, speed, wall time, and peak
   VRAM.

**Test:** CPU smoke runs prove exact initialization, incompatible checkpoints
are rejected, masked user tokens have zero gradient contribution, and output
runs are deterministic under the same seed.

**Commit:**

```text
feat: train CPT and SFT from declared checkpoints
```

## Task 4: Compare Base, CPT, and SFT on one frozen suite

**Files:**

- Create: `training/nanogpt_nspire/assistant_eval.py`
- Create: `tests/python/test_assistant_eval.py`
- Create: `experiments/lesson12-training.json`

**Steps:**

1. Encode assistant prompts as
   `<BOS><USER>question<ASSISTANT>` and generate only the continuation.
2. Stop at `<EOS>`, reject any non-byte special token in ordinary answer text,
   and keep context truncation deterministic.
3. Score exact arithmetic with a strict final-number parser, physics
   multiple-choice with a strict first-choice parser, answer termination, role
   leakage, repeated-phrase rate, and fixed greedy completions.
4. Evaluate all three checkpoints with the same prompts, maximum generated
   tokens, decoding rule, and tokenizer.
5. Report per-stage deltas without treating an SFT-formatted answer as proof
   of general intelligence.

**Test:** Hand-constructed outputs exercise correct, wrong, malformed,
unterminated, role-leaking, and repeated-answer cases.

**Commit:**

```text
test: compare Base CPT and SFT behavior
```

## Task 5: Run CUDA pilots, teach the result, and close regression gates

**Files:**

- Create: `docs/lessons/12-math-physics-cpt-and-sft.md`
- Modify: `README.md`

**Steps:**

1. Create an isolated Lesson 12 environment using the already validated CUDA
   package set.
2. Run short overfit gates before the bounded CPT and SFT pilots.
3. Select checkpoints only from validation metrics, then run the frozen test
   and qualitative suite once for the committed comparison.
4. Explain CPT versus SFT, checkpoint initialization, replay and catastrophic
   forgetting, assistant-only masking, why `<USER>`/`<ASSISTANT>` are now
   model-visible, and why a tiny model can learn format before robust
   reasoning.
5. Run the full Python suite and Host C CTest suite, scan tracked/generated
   metadata for credential-shaped values, commit, merge to `main`, and push.

**Test:** All Python tests and Host C tests pass; the committed experiment
records match ignored run files and checkpoint hashes; Git is clean and
`origin/main` equals local `main`.

**Commit:**

```text
docs: teach math physics CPT and role-aware SFT
```

## Frozen downstream order

1. **Lesson 13:** external sequence teacher plus shared-tokenizer local-logit
   distillation.
2. **Lesson 14:** direct-answer versus CoT SFT/RLVR under equal generated-token
   limits.
3. **Lesson 15:** real W4A8 export, PyTorch/Host C alignment, safe arithmetic
   tool comparison, and physical Nspire deployment.

