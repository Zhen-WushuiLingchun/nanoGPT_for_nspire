# Lesson 10 English Base Data Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 65-character Shakespeare-only data contract with a
deterministic byte/special-token protocol and an auditable English
math/physics corpus pipeline suitable for later base pretraining and SFT.

**Architecture:** A fixed `264`-token codec handles raw bytes plus role,
reasoning, tool, and padding markers. Canonical corpus records pass through
license eligibility, normalization, family-level splitting, deduplication, and
task verification before atomic binary shards and a hashed manifest are
written. The first committed experiment uses project-authored deterministic
arithmetic records as a smoke corpus; large public datasets remain external and
are referenced by a machine-readable pinned registry.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `decimal`,
`fractions`, `hashlib`, `json`, `urllib`), NumPy/PyTorch only where later
training needs them, pytest 8+, existing atomic-write and SHA-256 conventions.

**Security and license boundary:** API credentials are read only from
`DEEPSEEK_API_KEY` by a later client and never appear in source, manifests,
exceptions, or test fixtures. Unknown, non-commercial, share-alike, and
AI-prohibited sources are ineligible for the initial permissive mix. Raw
corpora and model-generated records stay under ignored `artifacts/`.

---

## Task 1: Freeze the byte and special-token protocol

**Files:**

- Create: `training/nanogpt_nspire/byte_tokenizer.py`
- Create: `tests/python/test_byte_tokenizer.py`

**Step 1: Write failing codec tests**

Cover:

```python
def test_all_bytes_round_trip() -> None:
    tokenizer = ByteTokenizer()
    payload = bytes(range(256))
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(payload)) == payload


def test_special_ids_are_frozen() -> None:
    assert BOS_ID == 256
    assert EOS_ID == 257
    assert USER_ID == 258
    assert ASSISTANT_ID == 259
    assert TOOL_ID == 260
    assert THINK_ID == 261
    assert FINAL_ID == 262
    assert PAD_ID == 263
    assert VOCAB_SIZE == 264
```

Also reject booleans, negative IDs, IDs at or above `264`, and special tokens
when decoding raw bytes. Test strict UTF-8 decode and a diagnostic rendering
mode that names special tokens without confusing them with literal text.

**Step 2: Run the focused test**

Run:

```powershell
python -m pytest tests/python/test_byte_tokenizer.py -q
```

Expected: import failure because the module does not exist.

**Step 3: Implement the minimal codec**

Use token ID equal to byte value for `0..255`. Keep the special-token table
immutable and validate every input without implicit integer coercion.

**Step 4: Run the focused test**

Expected: all tokenizer tests pass.

**Step 5: Commit**

```text
feat: add fixed byte and role token protocol
```

## Task 2: Serialize role-aware conversations and loss masks

**Files:**

- Modify: `training/nanogpt_nspire/byte_tokenizer.py`
- Create: `tests/python/test_conversation_format.py`

**Step 1: Write failing formatting tests**

Construct:

```python
turns = (
    ConversationTurn("user", "What is 12 * 7?"),
    ConversationTurn("assistant", "12 times 7 is 84."),
)
tokens, loss_mask = format_conversation(turns)
```

Assert exact serialization:

```text
<BOS><USER>question bytes<ASSISTANT>answer bytes<EOS>
```

Assert the loss mask is `1` only for assistant answer bytes and final `<EOS>`.
Test two-turn ordering, empty text rejection, invalid role order, non-string
content, embedded NUL acceptance, and context-limit rejection.

**Step 2: Run the focused test**

Expected: missing conversation types/functions.

**Step 3: Implement conversation formatting**

Use a frozen `ConversationTurn` dataclass. Require alternating user/assistant
turns starting with user and ending with assistant. Return immutable tuples of
equal length. Do not render role names as ordinary byte strings.

**Step 4: Run tokenizer and conversation tests**

Expected: both focused suites pass.

**Step 5: Commit**

```text
feat: encode role-aware SFT conversations
```

## Task 3: Add a public-source registry and license gate

**Files:**

- Create: `training/nanogpt_nspire/source_registry.py`
- Create: `experiments/lesson10-public-sources.json`
- Create: `tests/python/test_source_registry.py`

**Step 1: Write failing registry tests**

Test:

- source IDs, URLs, revisions, subsets, license IDs, and intended stages are
  required and reject unknown keys;
- permissive licenses are accepted by the initial mix;
- `unknown`, `CC-BY-NC-*`, `CC-BY-SA-*`, and an explicit
  `ai-training-prohibited` policy are rejected;
- duplicate source IDs fail;
- registry JSON round-trips with deterministic key ordering;
- no field name or value matches a credential pattern.

The committed registry includes:

```text
fineweb-edu
common-corpus
openwebmath
deepmind-mathematics
gsm8k
openmathinstruct-2
oasst1
deepseek-v4-pro-generated
```

It also records excluded OpenStax, SciQ, and OpenBookQA entries with explicit
reasons so future changes cannot silently admit them.

**Step 2: Run the focused test**

Expected: source-registry import failure.

**Step 3: Implement strict parsing and eligibility**

Keep license policy code separate from descriptive provenance. Unknown values
fail closed. Do not fetch network content in unit tests.

**Step 4: Run the focused test**

Expected: registry tests pass.

**Step 5: Commit**

```text
data: register auditable public training sources
```

## Task 4: Generate exactly verifiable arithmetic curriculum

**Files:**

- Create: `training/nanogpt_nspire/math_curriculum.py`
- Create: `tests/python/test_math_curriculum.py`

**Step 1: Write failing generator tests**

Test deterministic generation for a seed, different sequences for different
seeds, unique IDs, exact results, and canonical output. Include:

- integer `+`, `-`, `*`, and exact `/`;
- decimal addition/subtraction/multiplication using `decimal.Decimal`;
- one pair of parentheses;
- concise direct answer;
- concise worked answer;
- stable `family_id` shared by direct/CoT/paraphrase variants.

Example:

```python
example = ArithmeticExample.create(
    left=12,
    operator="*",
    right=7,
)
assert example.exact_answer == "84"
assert verify_arithmetic_example(example)
```

Reject division by zero, non-finite decimals, unsupported operators, excessive
digit counts, and any answer that does not match the exact reference.

**Step 2: Run the focused test**

Expected: missing module.

**Step 3: Implement without `eval`**

Generate an expression tree and compute through explicit operator functions.
Use `Fraction` or `Decimal` as appropriate. Store the reference operation,
operands, canonical expression, exact answer, difficulty, and seed.

**Step 4: Run the focused test**

Expected: all curriculum tests pass.

**Step 5: Commit**

```text
feat: generate exact arithmetic curriculum
```

## Task 5: Build canonical corpus records and family-level splits

**Files:**

- Create: `training/nanogpt_nspire/base_corpus.py`
- Create: `tests/python/test_base_corpus.py`

**Step 1: Write failing record and split tests**

Cover:

- strict canonical record schema;
- stable SHA-256 record and normalized-text fingerprints;
- family-level `train` / `validation` / `test` split;
- all variants of one family always choose the same split;
- exact duplicates with conflicting families fail before output;
- output order is independent of input iteration order;
- invalid UTF-8, empty text, missing provenance, or ineligible licenses fail;
- binary token and loss-mask packing uses explicit types;
- two identical builds produce byte-identical files and manifest.

Use `90/5/5` as the default family split. Split assignment is:

```python
bucket = int.from_bytes(
    sha256(f"{split_seed}:{family_id}".encode()).digest()[:8],
    "big",
) % 10_000
```

**Step 2: Run the focused test**

Expected: missing corpus module.

**Step 3: Implement atomic shard building**

For base records, write `<BOS> bytes <EOS>`. For conversation records, call the
Task 2 formatter and write an aligned `uint8` loss mask. Token files are
explicit little-endian `uint16`. The manifest includes:

- schema/tokenizer versions;
- split rule and seed;
- source registry hash;
- record/family/token counts;
- file sizes and SHA-256;
- deduplication and eligibility summaries.

Write all files to a temporary sibling directory and rename only after every
hash and count is complete.

**Step 4: Run focused and existing Lesson 01 tests**

Expected: no regression to the frozen Shakespeare pipeline.

**Step 5: Commit**

```text
feat: build deterministic English corpus shards
```

## Task 6: Add the Lesson 10 smoke command and evidence

**Files:**

- Create: `training/nanogpt_nspire/lesson10_data.py`
- Create: `tests/python/test_lesson10_data.py`
- Create: `experiments/lesson10-base-data.json`
- Modify: `.gitignore` only if a new artifact path is not already covered

**Step 1: Write failing CLI tests**

Test:

```powershell
python -m nanogpt_nspire.lesson10_data smoke `
  --output artifacts/lesson10-smoke `
  --seed 20260728 `
  --examples 256
```

Assert:

- the command emits a bounded JSON summary;
- train/validation/test all contain at least one family;
- rerunning into a clean directory is byte-identical;
- a non-empty destination fails unless `--replace` is explicit;
- `--replace` validates the exact target is within the requested output
  directory before replacing it;
- no environment variables or secrets enter output.

**Step 2: Implement the command**

Generate a balanced project-authored arithmetic smoke corpus, create direct and
worked-answer variants, call the canonical builder, and print hashes/counts.
Do not call DeepSeek or download a public corpus in this acceptance run.

**Step 3: Run the command twice**

Expected: identical manifest and shard hashes.

**Step 4: Record compact evidence**

Commit counts, hashes, split seed, tokenizer contract, and source-registry hash
to `experiments/lesson10-base-data.json`. Do not commit raw generated examples
or binary shards.

**Step 5: Commit**

```text
test: record Lesson 10 data smoke evidence
```

## Task 7: Teach the new data foundation and run regression gates

**Files:**

- Create: `docs/lessons/10-english-byte-tokenizer-and-corpus.md`
- Modify: `README.md`

**Step 1: Write the lesson**

Explain:

- why a real base model still predicts the next token;
- character, byte, BPE, and special-token differences;
- why `USER` / `ASSISTANT` must be model tokens rather than UI metadata;
- assistant-only loss masking;
- document-family splitting and contamination;
- data quality versus data quantity;
- exact verifiers and why fluent teacher outputs are not automatically gold;
- source licenses and the OpenStax exclusion;
- why Base/SFT/distillation/RLVR claims remain separate;
- the four equal-output-budget direct/CoT experiments.

**Step 2: Update project navigation**

Mark Lesson 10 as the completed data-foundation step, but state explicitly
that no English base checkpoint has been trained yet.

**Step 3: Run Python tests**

```powershell
python -m pytest tests/python -q
```

Expected: all existing and new Python tests pass.

**Step 4: Run C regression tests**

```powershell
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected: existing runtime/UI tests remain green. Lesson 10 does not yet change
the `.ngm` format or C runtime.

**Step 5: Audit repository hygiene**

Run:

```powershell
git diff --check
git status --short
git grep -n -I -E "sk-[A-Za-z0-9]{16,}|DEEPSEEK_API_KEY="
```

Expected: no credential value, no raw corpus, no checkpoint, and no generated
binary shard is tracked.

**Step 6: Commit**

```text
docs: teach byte tokenizer and corpus provenance
```

## Deferred follow-on plans

Lesson 10 does not silently expand into model training. After its gates pass:

1. **Lesson 11:** CUDA environment, architecture/file/RAM estimator, and English
   Base pilot training.
2. **Lesson 12:** continued math/physics pretraining and role-aware SFT.
3. **Lesson 13:** DeepSeek V4-Pro sequence generation, verification, and local
   shared-tokenizer teacher.
4. **Lesson 14:** direct versus CoT SFT/RLVR under equal output-token budgets.
5. **Lesson 15:** byte/special-token `.ngm` v2, W4A8 alignment, calculator tool,
   and physical Nspire deployment.
