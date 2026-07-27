# Lesson 08 C Runtime Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export all three frozen small-model routes to one bounded binary format and run numerically aligned incremental inference in portable C on Host and the Ndless toolchain.

**Architecture:** A Python exporter emits a CRC-protected little-endian tensor table and aligned payloads. A shared C11 loader exposes zero-copy views into one model blob; incremental Transformer operators use either FP32 matvec or direct packed-INT4/dynamic-INT8 matvec while sharing FP32 attention and KV cache.

**Tech Stack:** Python 3.13, PyTorch, pytest, C11, CMake/CTest, MSVC 2022, WSL, Ndless SDK r2022, Arm GNU Toolchain 14.3.

---

### Task 1: Freeze and test the binary format

**Files:**
- Create: `training/nanogpt_nspire/export_format.py`
- Create: `tests/python/test_export_format.py`
- Modify: `training/nanogpt_nspire/__init__.py`

**Step 1: Write failing header/table tests**

Cover constant sizes, little-endian encoding, CRC validation, alignment, tensor-ID
mapping, FP32/INT4 byte accounting and vocabulary encoding.

**Step 2: Run the focused test**

Run:

```powershell
python -m pytest tests/python/test_export_format.py -q
```

Expected: FAIL because the format module does not exist.

**Step 3: Implement the minimum format module**

Implement immutable header/tensor records, checked integer helpers, CRC32,
alignment and strict parse/serialize functions. Do not load checkpoints yet.

**Step 4: Add malformed tests**

Mutate magic, version, endian marker, CRC, file size, offsets and tensor IDs.
Every mutation must raise a bounded `ModelFormatError`.

**Step 5: Verify and commit**

Run the focused test and `git diff --check`.

Commit:

```powershell
git add training/nanogpt_nspire/export_format.py tests/python/test_export_format.py
git commit -m "feat: define portable model format"
```

### Task 2: Export frozen PyTorch artifacts

**Files:**
- Create: `training/nanogpt_nspire/export_model.py`
- Create: `tests/python/test_export_model.py`
- Modify: `pyproject.toml`

**Step 1: Write failing tiny-model export tests**

Test FP32 alias deduplication, fixed tensor order, exact tensor round trip,
INT4 packed/scales round trip, route metadata, file limit and manifest hashes.

**Step 2: Implement checkpoint validation and exporter**

Accept:

- legacy Lesson 05 Direct-Small checkpoint plus its `run.json`;
- Lesson 07 Distilled-Small;
- Lesson 07 formal Quantized-Small.

Reject failed/diagnostic teachers and unsupported configs. Write `.ngm` atomically
plus a JSON manifest.

**Step 3: Export real artifacts**

Run:

```powershell
python -m nanogpt_nspire.export_model `
  --checkpoint artifacts/lesson05-direct-small/direct_small_gpt.pt `
  --output artifacts/lesson08-export/direct-small.ngm
```

Repeat for Distilled-Small and Quantized-Small. Verify each file is below
`6,291,456 bytes`.

**Step 4: Verify and commit**

Run focused tests, full pytest, parse each real export and compare payloads to
the source artifact.

Commit:

```powershell
git add training/nanogpt_nspire/export_model.py tests/python/test_export_model.py pyproject.toml
git commit -m "feat: export unified nspire models"
```

### Task 3: Add the portable C loader

**Files:**
- Create: `CMakeLists.txt`
- Create: `runtime/include/ng_model.h`
- Create: `runtime/src/ng_crc32.c`
- Create: `runtime/src/ng_model.c`
- Create: `runtime/platform/host/model_inspect.c`
- Create: `tests/c/test_model_loader.c`

**Step 1: Write failing C loader tests**

Use a tiny committed/generated fixture. Test success, every malformed class,
memory cap rejection and exact tensor view offsets.

**Step 2: Implement checked parsing**

Read the file into one bounded blob, validate the complete header/table/CRC and
derive the arena byte requirement before exposing views.

**Step 3: Build and run**

Run:

```powershell
cmake -S . -B build/host -G "Visual Studio 17 2022" -A x64
cmake --build build/host --config Release
ctest --test-dir build/host -C Release --output-on-failure
```

**Step 4: Commit**

```powershell
git add CMakeLists.txt runtime tests/c
git commit -m "feat: load bounded nspire model files"
```

### Task 4: Implement and align FP32 incremental inference

**Files:**
- Create: `runtime/include/ng_runtime.h`
- Create: `runtime/src/ng_ops.c`
- Create: `runtime/src/ng_runtime.c`
- Create: `runtime/platform/host/run_model.c`
- Create: `training/nanogpt_nspire/alignment.py`
- Create: `tests/c/test_ops.c`
- Create: `tests/python/test_host_alignment.py`

**Step 1: Write scalar operator golden tests**

Cover matvec, LayerNorm, softmax, GELU, attention, residual and cache append.

**Step 2: Implement arena planning and one-token forward**

Allocate KV and scratch once. No allocation is allowed inside
`ng_runtime_forward_token`.

**Step 3: Add PyTorch CPU probes**

Export last-token logits and a 64-token greedy sequence. Add optional
intermediate dumps to identify the first mismatching block.

**Step 4: Close the FP32 gates**

Require max absolute error `<=2e-4`, RMSE `<=5e-5` and exact greedy tokens for
both Direct and Distilled artifacts.

**Step 5: Commit**

```powershell
git add runtime training/nanogpt_nspire/alignment.py tests
git commit -m "feat: align fp32 host c inference"
```

### Task 5: Implement the direct packed W4A8 path

**Files:**
- Create: `training/nanogpt_nspire/w4a8_reference.py`
- Create: `tests/python/test_w4a8_reference.py`
- Modify: `runtime/src/ng_ops.c`
- Modify: `runtime/src/ng_runtime.c`
- Modify: `tests/c/test_ops.c`
- Modify: `tests/python/test_host_alignment.py`

**Step 1: Freeze dynamic activation quantization tests**

Test zero groups, rounding boundaries, `[-127,127]`, group reuse and
INT32 accumulation.

**Step 2: Implement Python W4A8 reference**

It must use the same per-group formulas as C and produce alignment fixtures.

**Step 3: Implement C packed kernel**

Decode low-first nibbles inside the dot loop. Assert/report that no FP32 matrix
copy or parameter-sized scratch exists.

**Step 4: Close alignment and quality gates**

Align C to Python W4A8, then measure Tiny Shakespeare validation loss against
the Lesson 07 W4A32 reference. Preserve a failed gate rather than retuning it.

**Step 5: Commit**

```powershell
git add training runtime tests
git commit -m "feat: add direct packed w4a8 inference"
```

### Task 6: Add Ndless compile/link/package smoke

**Files:**
- Create: `Makefile`
- Create: `runtime/platform/ndless/main_ndless.c`
- Create: `runtime/platform/ndless/crt_compat.c`
- Create: `tools/ndless-env.sh`
- Create: `tests/device/runtime_smoke.c`

**Step 1: Add an explicit source list**

Never use a wildcard that could pull Host code into the device binary.

**Step 2: Compile portable sources with warnings**

Use C11, `-marm`, `-Os`, function/data sections and the installed SDK paths
provided through environment variables.

**Step 3: Link and package**

Create `dist/nanogpt-runtime-smoke.tns` with `genzehn` and `make-prg`.
Record size and imports. Do not call this a真机 run.

**Step 4: Commit**

```powershell
git add Makefile runtime/platform/ndless tools/ndless-env.sh tests/device
git commit -m "build: add ndless runtime smoke"
```

### Task 7: Record Lesson 08 evidence

**Files:**
- Create: `docs/lessons/08-c-runtime-and-pytorch-alignment.md`
- Create: `experiments/lesson08-export-and-host-c.json`
- Modify: `experiments/small-model-comparison.json`
- Modify: `README.md`

**Step 1: Record source-backed measurements**

Include export bytes/SHA, tensor storage, exact arena bytes, Host logits
errors, greedy matches, malformed counts, Host timing and Ndless package size.

**Step 2: Explain every operator**

Teach row-major weights, LayerNorm, QKV split, cache shapes, stable softmax,
GELU, tied head and W4A8 group arithmetic.

**Step 3: Preserve claim boundaries**

Host/ARM build evidence is not CX II execution. Leave device time and tracked
heap explicitly pending Lesson 09.

**Step 4: Final verification**

Run:

```powershell
python -m pytest -q
cmake --build build/host --config Release
ctest --test-dir build/host -C Release --output-on-failure
wsl make ndless-smoke
git diff --check
```

Commit:

```powershell
git add README.md docs/lessons experiments
git commit -m "docs: record lesson 08 c alignment"
```
