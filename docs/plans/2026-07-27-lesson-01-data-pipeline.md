# Lesson 01 Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the repository skeleton and a deterministic, tested character-token data pipeline for Tiny Shakespeare.

**Architecture:** Keep official nanoGPT read-only under `upstream/`. Put the new standard-library Python package under `training/`, generate all large data under ignored `artifacts/`, and describe every runnable step in a Chinese lesson. The pipeline pins the source SHA-256, uses a sorted vocabulary, writes little-endian `uint16` token files, and records a machine-readable manifest.

**Tech Stack:** Python 3.10+, standard library, pytest 8+, Markdown, Git

---

### Task 1: Create the tracked repository skeleton

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `training/nanogpt_nspire/__init__.py`
- Create: `training/configs/README.md`
- Create: `runtime/README.md`
- Create: `runtime/include/README.md`
- Create: `runtime/src/README.md`
- Create: `runtime/platform/host/README.md`
- Create: `runtime/platform/ndless/README.md`
- Create: `tools/README.md`
- Create: `experiments/README.md`

**Step 1: Add artifact and Python ignores**

Ignore `artifacts/`, build outputs, caches, virtual environments, local checkpoints, and generated `.tns` files. Do not add an ignore rule that hides source, experiment summaries, or `upstream/nanoGPT/`.

**Step 2: Add Python package metadata**

Configure setuptools to discover `nanogpt_nspire` below `training/`. Configure pytest with:

```toml
[tool.pytest.ini_options]
pythonpath = ["training"]
testpaths = ["tests/python"]
addopts = "-ra"
```

**Step 3: Add navigation documents**

The root README must link to the approved design and Lesson 01. The component READMEs must state their boundary without claiming unimplemented functionality.

**Step 4: Verify the skeleton**

Run:

```powershell
python -m pip install -e .
git status --short
python -m pytest --collect-only
```

Expected: the local package is installed in editable mode, the new skeleton is visible, and pytest exits successfully or reports that no tests exist yet.

### Task 2: Specify character-token behavior with failing tests

**Files:**
- Create: `tests/python/test_data.py`

**Step 1: Write vocabulary and round-trip tests**

```python
def test_vocabulary_is_sorted_and_round_trips():
    text = "cab\n"
    vocab = build_vocabulary(text)
    assert vocab == ("\n", "a", "b", "c")
    assert decode_tokens(encode_text(text, vocab), vocab) == text
```

Also require `encode_text` to raise `DatasetError` for characters outside the vocabulary and `decode_tokens` to reject invalid token IDs.

**Step 2: Write deterministic split and binary-format tests**

```python
def test_split_uses_floor_boundary():
    train, val = split_tokens([0, 1, 2, 3, 4], train_fraction=0.8)
    assert train == [0, 1, 2, 3]
    assert val == [4]

def test_pack_u16_is_little_endian():
    assert pack_u16_le([1, 0x0203]) == b"\x01\x00\x03\x02"
```

**Step 3: Write artifact and manifest tests**

Prepare a tiny UTF-8 fixture into a pytest temporary directory. Assert that:

- `train.bin`, `val.bin`, and `manifest.json` exist;
- the ordered vocabulary and token counts are exact;
- source and output SHA-256 values match the bytes on disk;
- running the preparation twice produces byte-identical files.

**Step 4: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/python/test_data.py -q
```

Expected: collection fails because `nanogpt_nspire.data` does not exist.

### Task 3: Implement the deterministic data module

**Files:**
- Create: `training/nanogpt_nspire/data.py`
- Test: `tests/python/test_data.py`

**Step 1: Implement pure tokenizer functions**

Implement:

```python
def build_vocabulary(text: str) -> tuple[str, ...]: ...
def encode_text(text: str, vocabulary: Sequence[str]) -> list[int]: ...
def decode_tokens(tokens: Iterable[int], vocabulary: Sequence[str]) -> str: ...
def split_tokens(tokens: Sequence[int], train_fraction: float = 0.9) -> tuple[list[int], list[int]]: ...
def pack_u16_le(tokens: Iterable[int]) -> bytes: ...
```

Reject empty text, duplicate vocabulary entries, unknown characters, invalid token IDs, invalid split fractions, unusable splits, and token IDs outside `uint16`.

**Step 2: Implement deterministic artifact generation**

`prepare_dataset(source_path, output_dir)` must:

1. read and strictly decode UTF-8 source bytes;
2. hash the exact source bytes;
3. build the sorted vocabulary and encode all characters;
4. make the fixed 90%/10% sequential split;
5. write `train.bin` and `val.bin` as little-endian `uint16`;
6. write a stable JSON manifest with schema version, source hash, vocabulary, split counts, dtype, file sizes, and file hashes.

Use replace-on-success temporary files so an interrupted write does not silently leave a valid-looking final artifact.

**Step 3: Add pinned fetch and CLI commands**

Pin:

```python
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42"
    "b0075e16b8fc2e86bf0ca57e2dc565ed"
)
```

Expose:

```powershell
python -m nanogpt_nspire.data fetch --output artifacts/raw/tinyshakespeare.txt
python -m nanogpt_nspire.data prepare `
  --input artifacts/raw/tinyshakespeare.txt `
  --output artifacts/data/tinyshakespeare
```

The fetch command must reject a hash mismatch. The prepare command must print a concise JSON summary.

**Step 4: Run tests to verify success**

Run:

```powershell
python -m pytest tests/python/test_data.py -q
```

Expected: all Lesson 01 tests pass.

### Task 4: Write the first Chinese lesson

**Files:**
- Create: `docs/lessons/01-tokenization-and-dataset.md`

**Step 1: Explain the concepts**

Explain:

- why a language model predicts the next token rather than storing sentences;
- the difference among characters, tokens, token IDs, vocabulary, input context, and targets;
- why sorted vocabulary and source hashes are reproducibility requirements;
- why the first pipeline uses `uint16-le` even though 65 characters fit in `uint8`;
- how the data pipeline connects to embeddings and the final C runtime.

**Step 2: Explain every public function**

Document the inputs, outputs, failure cases, and data flow of every public function in `training/nanogpt_nspire/data.py`.

**Step 3: Add runnable commands and expected evidence**

Record the pinned source facts:

- source bytes/tokens: `1,115,394`;
- vocabulary size: `65`;
- train tokens: `1,003,854`;
- validation tokens: `111,540`;
- source SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

Do not record output artifact hashes until the implementation has generated and verified them.

### Task 5: Run real-data acceptance and record evidence

**Files:**
- Modify: `docs/lessons/01-tokenization-and-dataset.md`

**Step 1: Fetch through the repository-owned CLI**

Run:

```powershell
python -m nanogpt_nspire.data fetch --output artifacts/raw/tinyshakespeare.txt
```

Expected: `1,115,394` bytes and the pinned source SHA-256.

**Step 2: Prepare through the repository-owned CLI**

Run:

```powershell
python -m nanogpt_nspire.data prepare `
  --input artifacts/raw/tinyshakespeare.txt `
  --output artifacts/data/tinyshakespeare
```

Expected: vocabulary `65`, train tokens `1,003,854`, validation tokens `111,540`.

**Step 3: Verify generated files independently**

Recompute SHA-256 and sizes from the files, parse `manifest.json`, and verify `artifacts/` remains ignored by Git.

**Step 4: Record the observed artifact hashes**

Add the actual `train.bin`, `val.bin`, and `manifest.json` hashes to the lesson as observed evidence, not as predicted values.

**Step 5: Run the complete test suite**

Run:

```powershell
python -m pytest -q
git diff --check
git status --short
```

Expected: every test passes, no whitespace errors, and no generated artifact is staged or visible as untracked.

### Task 6: Commit and push Lesson 01

**Step 1: Review the staged scope**

Run:

```powershell
git diff --stat
git diff --check
```

Expected: only the skeleton, implementation plan, Lesson 01, Python data module, and tests are included.

**Step 2: Commit**

```powershell
git add .gitignore README.md pyproject.toml training runtime tools tests experiments docs
git commit -m "feat: add deterministic character data pipeline"
```

**Step 3: Push and verify**

```powershell
git push origin main
git status --short --branch
git ls-remote origin refs/heads/main
```

Expected: local `main`, `origin/main`, and the GitHub remote point to the same commit and the worktree is clean.
