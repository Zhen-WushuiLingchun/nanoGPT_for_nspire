# English Math/Physics Assistant Design

> **Status (2026-07-28):** The user approved an English-only assistant for
> short arithmetic and introductory math/physics questions. The project will
> combine a from-scratch shared-tokenizer model ladder with external
> sequence-level distillation. Arithmetic will be evaluated both as a pure
> neural task and as a later deterministic C-tool enhancement. Preference
> optimization and RLVR remain downstream experiments, not substitutes for
> base pretraining or SFT.

## 1. Goal and honest capability boundary

Build a small but genuine English decoder-only language model that can:

- carry a short `USER` / `ASSISTANT` conversation;
- answer integer, decimal, parenthesized, and four-operation arithmetic;
- answer introductory school-level mathematics and physics questions;
- give a compact answer of one to four sentences, with a formula when useful;
- run as a packed W4A8 model in the existing Ndless application.

This is not a plan for a general ChatGPT replacement. The deployable model is
expected to remain around ten million parameters and to use a short context.
Success means reliable behavior on a frozen, narrow evaluation suite, not broad
world knowledge.

The work is split so each claim remains attributable:

```text
English base pretraining
  -> continued math/physics pretraining
  -> role-aware SFT
  -> external sequence distillation
  -> shared-tokenizer soft-logit distillation
  -> optional RLVR / preference experiments
  -> W4A8 export and Nspire deployment
```

No later stage may rewrite an earlier baseline result. Combined improvements
are reported separately.

## 2. Model ladder

All locally trained models share one tokenizer and one transcript protocol:

| Model | Intended shape | Approximate role |
|---|---|---|
| Base/Student | 6 layers, 6 heads, width 384, context 256 | deployable 10–12M model |
| Local Teacher | 12 layers, 10 heads, width 640, context 256 | 55–65M computer-only teacher |
| External Teacher | DeepSeek V4-Pro | data generation, critique, sequence distillation |

The exact local shapes are provisional until the parameter/file/RAM estimator
passes. The deployable student must remain below the existing 6 MiB model-file
class after real packed W4A8 export. Context RAM, arena scratch, and output
projection cost are measured rather than inferred from file size.

The local teacher shares the student's tokenizer, so it can provide aligned
per-token logits for KL distillation. DeepSeek uses a different tokenizer and
therefore cannot be treated as a soft-logit teacher. It provides:

- candidate questions and concise solutions;
- critique and correction passes;
- reasoning traces stored separately from final answers;
- sequence-level targets that the local models learn with ordinary token loss.

Current DeepSeek API model names are `deepseek-v4-pro` and
`deepseek-v4-flash`. Dataset gold generation and physics review use V4-Pro.
Flash may later be tested for inexpensive paraphrase generation, but Flash
outputs never enter the gold split without the same verifier gates.

## 3. Byte tokenizer and special-token contract

The Shakespeare vocabulary is replaced by a fixed byte vocabulary:

```text
0..255  raw byte values
256     <BOS>
257     <EOS>
258     <USER>
259     <ASSISTANT>
260     <TOOL>
261     <THINK>
262     <FINAL>
263     <PAD>
```

`vocab_size = 264` is frozen before base training. UTF-8 text round-trips
through raw bytes, so the tokenizer has no unknown token. The data policy is
English-only, but retaining all byte values makes URLs, units, mathematical
symbols, and future source text representable without changing trained tensor
shapes.

Base documents serialize as:

```text
<BOS> document UTF-8 bytes <EOS>
```

SFT conversations serialize as:

```text
<BOS>
<USER> question bytes
<ASSISTANT> answer bytes
<EOS>
```

Only assistant content, `<FINAL>` content when present, and the terminating
`<EOS>` contribute to the default SFT loss. User text and padding receive an
ignore index. This makes roles part of the sequence the model learns, rather
than UI-only metadata.

CoT experiments use:

```text
<ASSISTANT><THINK> reasoning <FINAL> concise answer <EOS>
```

Reasoning tokens count against the same output-token and latency budget as
direct answers. The UI may hide `<THINK>` text in a later experiment, but hidden
display does not make its computation free.

The current `.ngm` vocabulary stores one Unicode character per token. A later
format revision must store byte/special-token metadata and IDs explicitly.
Lesson 10 intentionally does not pretend the old exporter already supports the
new protocol.

## 4. Public data recipe and licensing

Raw large corpora remain outside Git. Every accepted source has an immutable
source record with repository, revision, subset, upstream license, retrieval
date, transformations, and content hashes.

### 4.1 Base and continued-pretraining sources

| Source | Planned use | License posture |
|---|---|---|
| FineWeb-Edu | high-score English educational prose | ODC-By 1.0 |
| Common Corpus | traceable English public-domain/CC-BY documents | per-document license retained |
| OpenWebMath | cleaned mathematical prose and notation | ODC-By 1.0 |
| Generated arithmetic/physics text | exact narrow-domain curriculum | project-authored MIT data/code |

FineWeb-Edu is streamed and deterministically sampled; the project does not
download the trillion-token corpus. Common Corpus is accepted only when the
row license is explicitly public domain, CC0, or attribution-only. Unknown,
non-commercial, and share-alike rows are excluded from the first permissive
training mix. OpenWebMath is filtered for document length, readable equations,
and the 256-token curriculum.

### 4.2 SFT and evaluation sources

| Source | Planned use | License |
|---|---|---|
| DeepMind Mathematics Dataset | generated curriculum and exact-answer tests | Apache-2.0 |
| GSM8K | human-written word-problem SFT; original test held out | MIT repository |
| OpenMathInstruct-2 | verified short math solutions only | CC-BY-4.0 |
| OpenAssistant OASST1 | small high-rated English dialogue subset | Apache-2.0 |
| DeepSeek V4-Pro outputs | physics explanations and distillation targets | generated-data provenance retained |

OpenStax College Physics 2e is excluded. Its current preface explicitly states
that the text may not be ingested into generative-AI offerings without
permission. SciQ is excluded from the permissive training mix because its
dataset card is CC-BY-NC-3.0. OpenBookQA is excluded while its public dataset
card reports an unknown license.

DeepSeek's current terms permit outputs to be used to train or distill other
models. Every published synthetic record must still be marked AI-generated,
verified, and linked to its prompt/model version. The API key is read only from
`DEEPSEEK_API_KEY`; it is never printed, serialized, cached in a manifest, or
committed.

## 5. Quality, split, and contamination gates

An example is eligible only after:

1. strict UTF-8 validation;
2. English/source-license eligibility;
3. bounded byte/token length;
4. exact and normalized duplicate removal;
5. removal of document boilerplate and malformed control bytes;
6. source and transformation provenance;
7. split assignment by stable content-family hash;
8. task-specific verification.

Splits occur before paraphrase or teacher expansion. A seed problem, all of its
paraphrases, its reasoning variants, and its tool form share one family ID and
must remain in one split. This prevents nearly identical generated problems
from leaking into validation or test.

Arithmetic uses `int`, `decimal.Decimal`, or `fractions.Fraction` reference
evaluation—never Python `eval`. Physics numerical examples carry:

- named formula;
- SI quantity values and units;
- exact or tolerance-based expected result;
- dimensional/unit check;
- deterministic generator seed.

Conceptual physics has two independent checks: a constrained V4-Pro critic
pass and a frozen human-readable gold record. Disagreement is quarantined, not
resolved by silently trusting the teacher.

Held-out evaluation data is never passed to DeepSeek after it is frozen. The
manifest records prompt templates and corpus hashes so benchmark
contamination can be audited.

## 6. Training stages and fairness

### Stage A: base pretraining

Train from random initialization on approximately 256M byte tokens for the
student pilot. Validation uses held-out documents, not random windows from the
same document. The local teacher first runs a smaller compute pilot, then
targets roughly 1B tokens only if loss and downstream probes justify it.

### Stage B: continued pretraining

Continue the selected base checkpoint on a controlled math/physics prose mix.
Keep a replay fraction of general English to measure and limit catastrophic
forgetting.

### Stage C: SFT and distillation

Freeze the base checkpoint and create:

- role-aware SFT;
- external sequence-distilled SFT;
- local soft-logit distillation;
- combined sequence + logits experiment.

Each comparison records architecture, initialization, optimizer tokens,
dataset hash, answer-token loss mask, and decoding policy.

### Stage D: pure neural versus tool-assisted arithmetic

The pure model answers without tools. The hybrid route sends a recognized
expression to a safe C parser and formats its exact result. Tool use is an
enhancement experiment and cannot replace the pure-model score.

### Stage E: RLVR and CoT

Only after Base and SFT gates pass, compare:

1. direct-answer SFT;
2. CoT-SFT;
3. direct-answer RLVR;
4. CoT-RLVR.

All start from declared checkpoints and use the same architecture, evaluation
questions, maximum generated tokens, and decoding rule. Report final-answer
accuracy, invalid format rate, reasoning length, TTFT, decode speed, and total
generated tokens. Reward is computed from a strict final-answer parser and
exact/tolerance verifier; no reward is assigned to fluent but wrong reasoning.

## 7. Data flow and artifacts

```text
source registry
  -> pinned downloader/stream adapter
  -> license gate
  -> normalization and quality filters
  -> family-level split
  -> task verifier
  -> canonical JSONL shards
  -> byte/special-token encoder
  -> binary training shards + manifest
```

Committed files include code, compact test fixtures, source registries,
experiment configs, manifests without raw examples, and lessons. Downloaded
corpora, teacher prompts/responses, token shards, checkpoints, and API billing
logs stay under ignored `artifacts/`.

Every build is atomic: write temporary files, flush, hash, then rename. A
partial download or API run remains resumable but is never accepted as a
complete dataset.

## 8. Error handling and cost controls

- Missing API key: fail before creating output files.
- Authentication error: redact headers and response bodies that may echo
  credentials.
- Rate limit/server error: bounded exponential retry with a request ID.
- Invalid/empty JSON: quarantine the response with non-secret metadata.
- Token/cost budget reached: stop cleanly and write an incomplete run summary.
- License missing or disallowed: reject before content enters training.
- Verifier disagreement: quarantine rather than majority-vote into gold data.
- Hash mismatch: delete only the incomplete temporary file and require an
  explicit source revision update.

No API batch is launched without a dry-run count and estimated token/cost
ceiling.

## 9. Test and acceptance strategy

Lesson 10 acceptance is deliberately smaller than “train the final model”:

- all 256 bytes and all special tokens round-trip;
- role formatting and assistant-only loss masks are exact;
- stable family IDs keep variants in one split;
- license gates reject unknown, NC, SA, and AI-prohibited sources;
- arithmetic generation is deterministic and exactly verified;
- duplicate records cannot cross splits;
- a smoke corpus builds byte-identical shards and manifests twice;
- no secret-like value appears in committed or generated metadata.

Later model gates add base loss/BPC, fixed completion probes, SFT task accuracy,
RLVR reward curves, PyTorch/C alignment, packed file size, Host arena, and
physical Nspire speed/RAM.

## 10. Primary source records

- FineWeb-Edu:
  <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- Common Corpus:
  <https://huggingface.co/datasets/PleIAs/common_corpus>
- OpenWebMath:
  <https://huggingface.co/datasets/open-web-math/open-web-math>
- DeepMind Mathematics Dataset:
  <https://github.com/google-deepmind/mathematics_dataset>
- GSM8K:
  <https://github.com/openai/grade-school-math>
- OpenMathInstruct-2:
  <https://huggingface.co/datasets/nvidia/OpenMathInstruct-2>
- OpenAssistant OASST1:
  <https://huggingface.co/datasets/OpenAssistant/oasst1>
- DeepSeek V4 models and thinking mode:
  <https://api-docs.deepseek.com/quick_start/pricing/>
  and <https://api-docs.deepseek.com/guides/thinking_mode>
- DeepSeek Terms of Use:
  <https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html>
- OpenStax exclusion evidence:
  <https://openstax.org/books/college-physics-2e/pages/preface>
