# Lesson 09: Ndless Pixel Chat UI Implementation Plan

> **Status (2026-07-28):** Tasks 1-6 and the Host/ARM parts of Task 7 are
> complete. Physical upload/readback, launch, high-resolution timing, and
> calculator-side exit verification remain pending after a screen-off LibUSB
> interruption. Do not treat package success as device execution evidence.

> **For Codex:** Follow this plan in order. Keep the portable state machine and
> renderer independent from Ndless so every privacy and layout invariant can be
> tested on the host before packaging the calculator program.

**Goal:** Build a native 320x240 Ndless conversation shell around the Lesson 08
incremental C runtime, with fixed-capacity memory, one-token-per-loop decoding,
visible performance telemetry, and verified no-persistence cleanup.

**Architecture:** A portable `ng_chat` controller owns input, transcript cells,
prefill tokens, generation state, and telemetry without allocating memory. A
portable RGB565 renderer turns that state into a deterministic framebuffer. The
Host executable renders fixtures and exercises the real `.ngm` session; the
Ndless adapter only owns LCD/key/time/file integration and the model/runtime
allocations. This keeps the UI testable while making the calculator-specific
surface small and auditable.

**Tech Stack:** C11, CMake/CTest, Python test suite, RGB565 framebuffer, Ndless
r2022 SDK, `nspire-gcc`, `genzehn`, `make-prg`.

**License boundary:** The adjacent NspirePhysics project is GPL-3.0. It may be
consulted for installed-SDK behavior, but no source or font data is copied. New
code in this repository remains MIT. The bitmap font must be independently
authored or vendored from an explicitly compatible source with attribution.

---

## Task 1: Fixed-capacity conversation state

**Files:**

- Create: `runtime/include/ng_chat.h`
- Create: `runtime/src/ng_chat.c`
- Create: `tests/c/test_chat.c`
- Modify: `CMakeLists.txt`

**Step 1: Write failing tests**

Cover:

- inserting, moving, and deleting text in the input line;
- rejecting empty submissions and unsupported bytes;
- appending bounded USER/AI cells without heap allocation;
- deterministic overflow behavior for transcript and input pools;
- `New Chat` resetting transcript, input, token counters, and runtime context;
- `Shutdown` zeroing private buffers and detaching model/runtime pointers.

**Step 2: Run the focused test**

Expected: compilation fails because `ng_chat` does not exist.

**Step 3: Implement the minimal state machine**

Use fixed arrays only. Separate phases `IDLE`, `PREFILL`, `GENERATING`, `DONE`,
and `ERROR`. Preserve exact transcript bytes even when the display font falls
back for unsupported glyphs.

**Step 4: Run focused and existing C tests**

Expected: all pass.

**Step 5: Commit**

```text
feat: add fixed-capacity chat state
```

## Task 2: Incremental model session and telemetry

**Files:**

- Modify: `runtime/include/ng_chat.h`
- Modify: `runtime/src/ng_chat.c`
- Create: `tests/c/test_chat_session.c`
- Modify: `CMakeLists.txt`

**Step 1: Write failing integration tests**

Load the deterministic tiny FP32 fixture and assert:

- submit prepares a bounded prefill queue;
- each `ng_chat_step` consumes at most one model token;
- generation is deterministic under greedy decoding;
- context usage never exceeds `block_size`;
- first-token latency and decode-rate fields change only at the documented
  boundaries;
- cancelling and resetting clears the KV cache through `ng_runtime_reset`.

**Step 2: Implement token lookup and session stepping**

Map UTF-8 input scalars to the model vocabulary by exact byte match. Feed one
prefill token or generate one token per call. Use newline as a separator only
when present in the vocabulary. Stop at the configured token limit, context
limit, repeated newline, user cancellation, or runtime error.

**Step 3: Run focused and full C tests**

Expected: deterministic pass with no extra allocation inside a step.

**Step 4: Commit**

```text
feat: drive incremental generation from chat state
```

## Task 3: Portable pixel renderer

**Files:**

- Create: `runtime/include/ng_gfx.h`
- Create: `runtime/include/ng_chat_view.h`
- Create: `runtime/src/ng_gfx.c`
- Create: `runtime/src/ng_chat_view.c`
- Create: `runtime/src/ng_font.h`
- Create: `tests/c/test_chat_view.c`
- Modify: `CMakeLists.txt`

**Step 1: Write failing rendering tests**

Assert clipping, RGB565 fills/lines, text bounds, deterministic framebuffer
hashes, cell wrapping, scroll clamping, and footer values at 320x240.

**Step 2: Implement the renderer**

Use a deliberately compact industrial-terminal visual system:

- graphite/navy background;
- phosphor-mint AI cells;
- amber USER cells and coral warnings;
- sharp one-pixel borders and a bitmap font;
- fixed title, transcript, input, and telemetry regions.

The renderer must never allocate and must clip every draw operation.

**Step 3: Run focused tests**

Expected: deterministic hashes on Windows and Linux/WSL builds.

**Step 4: Commit**

```text
feat: render pixel chat interface
```

## Task 4: Host fixture and visual QA

**Files:**

- Create: `runtime/platform/host/chat_fixture.c`
- Modify: `CMakeLists.txt`
- Create: `experiments/lesson09-chat-ui.json`

**Step 1: Render representative states**

Produce lossless fixtures for:

- empty ready state;
- multi-cell conversation;
- active generation with telemetry;
- context-full/error warning.

**Step 2: Convert and inspect**

Render the PPM output to PNG, inspect it at original resolution, and revise any
clipping, low-contrast, or hierarchy problems.

**Step 3: Record deterministic evidence**

Record framebuffer hashes, dimensions, tracked static bytes, and fixture paths.

**Step 4: Commit**

```text
test: add headless chat visual fixtures
```

## Task 5: Ndless platform adapter

**Files:**

- Create: `runtime/platform/ndless/chat_platform_ndless.h`
- Create: `runtime/platform/ndless/chat_platform_ndless.c`
- Create: `runtime/platform/ndless/chat_main_ndless.c`
- Modify: `Makefile`

**Step 1: Implement explicit lifecycle ownership**

The adapter owns:

- RGB565 backbuffer allocation;
- `.ngm.tns` model loading and runtime arena allocation;
- LCD mode switch and restoration;
- key edge detection and text mapping;
- tracked live/peak allocation counters.

No allocation is permitted in the token loop.

**Step 2: Bind controls**

- letters/digits/punctuation: edit input;
- arrows: input cursor or transcript scroll;
- Enter: submit;
- Esc while generating: cancel;
- Menu: new chat and zero buffers/KV;
- Ctrl+Esc: exit and release everything.

**Step 3: Build ARM package**

Add `make ndless-chat`, keep `ndless-smoke`, and package
`dist/nanogpt-chat.tns`.

**Step 4: Audit the binary**

Record ELF section sizes, package bytes, unresolved symbols, and confirm both
portable and Ndless sources compile with warnings as errors.

**Step 5: Commit**

```text
feat: package native Ndless chat shell
```

## Task 6: End-to-end regression and privacy gate

**Files:**

- Modify: `tests/c/test_chat.c`
- Modify: `tests/c/test_chat_session.c`
- Modify: `CMakeLists.txt`
- Modify: `experiments/lesson09-chat-ui.json`

**Step 1: Add lifecycle probes**

Run repeated submit/generate/new-chat cycles. Verify stable tracked memory and
that every private byte is zero after reset/shutdown.

**Step 2: Run all gates**

```text
python -m pytest
cmake --build build/host --config Release
ctest --test-dir build/host -C Release --output-on-failure
make ndless-smoke
make ndless-chat
```

**Step 3: Keep claims bounded**

Host/ARM build success is not calculator launch proof. Do not record true
tokens/s, peak RAM, or display restoration as measured until the physical CX II
run is performed.

**Step 4: Commit**

```text
test: verify chat lifecycle and ARM package
```

## Task 7: Lesson 09 and handoff

**Files:**

- Create: `docs/lessons/09-ndless-chat-ui.md`
- Modify: `README.md`
- Modify: `docs/plans/2026-07-28-lesson-09-ndless-chat-ui-design.md`
- Modify: `experiments/lesson09-chat-ui.json`

**Step 1: Explain the learning line**

Teach:

- why prompt prefill and autoregressive decode are separate states;
- how a sequence-completion model can sit behind a chat-shaped UI without
  becoming an instruction-following model;
- why KV/context reset is a privacy and correctness requirement;
- why model bytes, arena, framebuffer, UI pools, and allocator overhead are
  separate memory terms;
- why Host timing, yield-count timing, and hardware timing are not equivalent.

**Step 2: Document deployment**

Describe the app and model filenames, calculator document directory, key map,
expected startup errors, and clean-exit behavior.

**Step 3: Record remaining physical gate**

Create a short checklist for launch, multi-turn input, cancellation, new-chat
zeroing, exit/relaunch, measured speed, and observed peak memory on the actual
CX II CAS.

**Step 4: Run final validation and push**

Commit the lesson/evidence, push `main`, then confirm local `HEAD` equals
`origin/main`.
