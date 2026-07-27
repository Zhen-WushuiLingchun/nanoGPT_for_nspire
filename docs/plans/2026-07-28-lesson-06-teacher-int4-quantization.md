# Lesson 06 Teacher and INT4 Quantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train and quality-gate the provisional 6×384 teacher, then convert its unique matrix weights to a deterministic packed groupwise INT4 representation and measure quality/size loss without claiming the future C integer runtime already exists.

**Architecture:** Reuse the complete GPT and deterministic training engine from Lesson 05 with an explicit run identity for the teacher. Freeze teacher context 128, 6 layers, 6 heads, width 384, dropout 0.2, and 81,920,000 training tokens. Quantize every unique 2D parameter symmetrically along its last dimension in groups of 64 to signed INT4 with FP32 scales; preserve 1D LayerNorm weights in FP32 and serialize tied token/head weights once. Evaluate a reconstructed FP32 reference to isolate weight quantization error while retaining a separate pending gate for W4A8/int32 C execution.

**Tech Stack:** Python 3.10+, PyTorch 2+, pytest 8+, CUDA, JSON, Markdown

---

### Task 1: Make the existing trainer identify different routes

**Files:**
- Modify: `training/nanogpt_nspire/direct_small_train.py`
- Modify: `tests/python/test_direct_small_train.py`

**Step 1: Specify a run identity**

Add an immutable record containing:

```text
route
checkpoint filename
optional maximum selected validation loss
deployment interpretation
```

The default identity must reproduce the existing Direct-Small behavior and
`direct_small_gpt.pt` filename. Reject empty routes, unsafe/non-basename
filenames, non-`.pt` suffixes, and invalid quality thresholds.

**Step 2: Record quality-gate results**

When a maximum loss is supplied, add:

```text
quality_gate_maximum_selected_validation_loss
quality_gate_passed
```

to the summary and checkpoint. The trainer still preserves a failed checkpoint
for diagnosis.

**Step 3: Verify Direct-Small regression**

Run existing Direct-Small tests and require all prior summary fields, schedule,
optimizer grouping, and default checkpoint filename to remain unchanged.

### Task 2: Add the frozen teacher training entry

**Files:**
- Create: `training/nanogpt_nspire/teacher_train.py`
- Create: `tests/python/test_teacher_train.py`

**Step 1: Freeze the profile**

Default teacher:

```text
vocab_size     = dataset vocabulary
block_size     = 128
n_layer        = 6
n_head         = 6
n_embd         = 384
mlp_ratio      = 4
dropout        = 0.2
bias           = false
tie_embeddings = true
steps          = 10,000
batch_size     = 64
training token = 81,920,000
```

Use the Lesson 05 AdamW, warmup/cosine, validation and sample protocol. Set:

```text
route = Teacher
checkpoint = teacher_gpt.pt
quality gate loss <= 1.4797899746894836
```

**Step 2: Test the exact parameter formula**

Require:

```text
parameters       = 10,695,936
raw FP32 bytes   = 42,783,744
ideal INT4 lower bound = 5,347,968 bytes
```

The FP32 teacher itself is not deployment-file eligible; that is expected.

**Step 3: Add a CPU smoke run**

Allow a small test config while retaining the Teacher run identity. Require the
teacher checkpoint name, route, quality gate, source commit, and strict reload.

### Task 3: Commit and train the teacher

**Files:**
- Create at runtime: `artifacts/lesson06-teacher/teacher_gpt.pt`
- Create at runtime: `artifacts/lesson06-teacher/run.json`

**Step 1: Verify and commit trainer code**

Run focused and full tests, compile, check CLI help and `git diff --check`, then
commit:

```text
feat: add provisional teacher training profile
```

**Step 2: Run from the exact commit**

Train with the frozen profile and source commit. Preserve all artifacts whether
the quality gate passes or fails.

**Step 3: Independently reproduce**

Strictly reload and require:

- unique parameter count and tied identity;
- fixed-window selected validation loss exact match;
- fixed-seed sample exact match;
- causal future isolation;
- source commit, artifact bytes, and SHA-256;
- quality gate evaluated against the preregistered threshold.

Stop the quantized route if the teacher fails the quality gate. Quantization
utilities may still be tested pedagogically, but the failed model cannot become
the official Quantized-Small source.

### Task 4: Specify signed groupwise INT4

**Files:**
- Create: `training/nanogpt_nspire/quantization/int4.py`
- Create: `training/nanogpt_nspire/quantization/__init__.py`
- Create: `tests/python/test_int4_quantization.py`

**Step 1: Test nibble packing**

Pack two signed values per byte in low-nibble-first order using two's-complement
codes. Require exact round trips for all values `[-8,7]`, odd lengths, padding,
and malformed metadata rejection.

**Step 2: Test symmetric group quantization**

For each last-dimension group:

```text
scale = max(abs(weight_group)) / 7
q = clamp(round(weight / scale), -7, 7)
```

Zero groups use scale `1.0` and all-zero codes. Require reconstruction error no
larger than approximately half a scale, finite FP32 scales, deterministic bytes,
and support for a final partial group.

**Step 3: Track physical bytes**

Record separately:

```text
packed nibble bytes
FP32 scale bytes
FP32 passthrough bytes
logical payload bytes
metadata reserve
actual torch artifact bytes
```

Do not count a duplicated tied alias.

### Task 5: Quantize and reconstruct a complete GPT state

**Files:**
- Create: `training/nanogpt_nspire/quantization/model_state.py`
- Create: `tests/python/test_quantized_model_state.py`

**Step 1: Canonicalize aliases**

Use `named_parameters(remove_duplicate=False)` to identify shared Parameters.
Store the first name as canonical and record later names in an alias table.

**Step 2: Apply the frozen storage policy**

- all unique 2D tensors: groupwise signed INT4, group size 64, FP32 scales;
- all unique 1D tensors: FP32 passthrough;
- token embedding/lm head: one physical tensor plus alias;
- no derived causal masks.

Reject unexpected tensor ranks or schema inconsistencies.

**Step 3: Reconstruct for quality evaluation**

Instantiate the original `DirectSmallGPT`, dequantize each canonical tensor, map
aliases, and load strictly. This path is explicitly named the dequantized PyTorch
reference; it is not presented as integer inference.

### Task 6: Add the teacher quantization experiment CLI

**Files:**
- Create: `training/nanogpt_nspire/quantize_teacher.py`
- Create: `tests/python/test_quantize_teacher.py`

**Step 1: Validate the source checkpoint**

Require:

- model type `direct_small_gpt`;
- teacher route and passed quality gate;
- frozen 6×384 architecture;
- tied vocabulary head;
- vocabulary and source hashes;
- selected validation loss no greater than the teacher gate.

**Step 2: Quantize and save**

Write:

```text
artifacts/lesson06-quantized/
├── teacher_int4.pt
└── run.json
```

The package contains only packed tensors, scales, FP32 passthrough vectors,
aliases, config, vocabulary and provenance—not the original FP32 matrices.

**Step 3: Evaluate quality**

Using fixed validation windows and sample seed, record:

- FP32 teacher loss/BPC;
- dequantized INT4 loss/BPC;
- absolute/relative loss degradation;
- fixed-probe logits max absolute error and RMSE;
- tensor-wise max absolute error and RMSE;
- fixed-seed quantized sample;
- logical and actual artifact sizes.

Pre-register:

```text
INT4 validation-loss degradation <= 0.05
logical payload + 64 KiB metadata <= 6 MiB
```

**Step 4: Preserve runtime boundary**

Record the route as:

```text
packed_weight_and_dequantized_reference_complete
integer_C_runtime = pending Lesson 08
Nspire measurement = pending Lesson 09
```

It cannot be marked final Quantized-Small until the C runtime directly consumes
packed INT4 weights without whole-model FP32 expansion.

### Task 7: Teach quantization and record results

**Files:**
- Create: `docs/lessons/06-int4-quantization.md`
- Create: `experiments/lesson06-teacher.json`
- Create: `experiments/lesson06-int4.json`
- Modify: `experiments/small-model-comparison.json`
- Modify: `README.md`

Explain:

- bits, signed ranges, scale and rounding;
- per-tensor versus per-channel/group trade-offs;
- two nibbles per byte;
- why symmetric INT4 uses `[-7,7]` while the encoding supports `-8`;
- padding and group metadata;
- FP32 LayerNorm exception;
- tied-weight deduplication;
- quantization error versus integer-kernel error;
- why a packed checkpoint is not yet proof of true integer Nspire inference.

Update the comparison state only with measured facts. If teacher or INT4 quality
fails, keep the route failed/candidate and do not promote it.

### Task 8: Final verification and push

Strictly reload both teacher and INT4 artifacts, reproduce metrics/sample,
cross-check committed JSON against ignored runs and hashes, run the full suite,
compile, verify artifacts are ignored, commit bounded evidence, push, and verify
local/tracking/remote `main` agree.
