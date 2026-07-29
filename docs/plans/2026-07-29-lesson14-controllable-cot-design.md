# Lesson 14: Controllable Short-CoT SFT and Context Extension Design

## Goal

Test whether explicit, short, verifiable chain-of-thought supervision improves
the frozen math/physics exact-answer gate, and whether one 10.8M-parameter
student can switch between direct and reasoning-first output through a learned
assistant-prefix cue.

This lesson is SFT only. RLVR remains a later experiment.

## Claim boundary

Lesson 14 may establish:

- whether short-CoT SFT changes exact accuracy under a fixed generation budget;
- whether a paired SFT corpus makes `<THINK>` and `<FINAL>` useful mode cues;
- how often CoT loses only because the fixed output budget truncates it;
- the parameter and incremental KV-arena cost of extending 256 to 512 tokens;
- whether a bounded 512-token continuation pilot can train without changing the
  tokenizer.

It may not establish:

- that visible reasoning is faithful to an internal computation;
- that prompt wording alone creates reasoning capability;
- that a 512-token checkpoint already runs in the current NGM/C/Nspire path;
- that DeepSeek-scale results transfer to a 10.8M-parameter byte model;
- any RLVR, preference-optimization, or quantized-device result.

## Why this is still SFT

Each target sequence is a fixed supervised demonstration. The optimizer uses
assistant-only cross entropy. No sampled completion receives a reward, no
policy ratio is computed, and no verifier participates in an online policy
update.

The DeepSeek-R1 work motivates cold-start reasoning demonstrations before RL,
but it does not make our supervised demonstrations RL. DeepSeek-V3.1 also
shows that a single post-trained model can expose thinking and non-thinking
modes through different chat-template prefixes. We copy the experimental idea,
not the model scale or its private training recipe.

Primary references:

- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning*, arXiv:2501.12948.
- DeepSeek-AI, *DeepSeek-V3.1* model card and official chat-template contract.
- Press, Smith, and Lewis, *Train Short, Test Long: Attention with Linear
  Biases Enables Input Length Extrapolation*, arXiv:2108.12409.
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language
  Models*, arXiv:2309.00071.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints*, arXiv:2305.13245.
- DeepSeek-AI, *DeepSeek-V2*, arXiv:2405.04434.
- Beltagy, Peters, and Cohan, *Longformer*, arXiv:2004.05150.

## Mode protocol

The 264-token vocabulary already reserves:

```text
261 <THINK>
262 <FINAL>
```

No tokenizer change is required.

Direct mode:

```text
<BOS><USER>question<ASSISTANT><FINAL>answer<EOS>
```

`<FINAL>` is supplied as the assistant prefix. The model generates only the
answer and `<EOS>`.

Reasoning mode:

```text
<BOS><USER>question<ASSISTANT><THINK>short reasoning<FINAL>answer<EOS>
```

`<THINK>` is supplied as the assistant prefix. The model must generate the
reasoning, the `<FINAL>` transition, the final answer, and `<EOS>`.

This is intentionally stronger than relying on a natural-language system
prompt. A system prompt is only another token sequence. It becomes a reliable
control only when its wording and both desired modes are represented in
training. A dedicated prefix token makes the independent variable exact and
costs one context position.

## Supervised data

Only public or project-authored, exactly verifiable records are eligible:

- project arithmetic: deterministic expression and worked answer;
- project introductory physics: formula, explicit substitution, value, unit;
- GSM8K training split: public worked rationale, stripped of annotation
  markup, with the official final decimal independently parsed;
- no GSM8K test row and no frozen Lesson 12 evaluation family.

The provider's private `reasoning_content` is not used. Lesson 13 synthetic
final text is also excluded from the primary comparison so that the change in
Lesson 14 is attributable to output structure rather than a new provider call.

All direct and CoT variants of one problem share a `family_id` and therefore a
train/validation/test split.

## Fair comparison

Primary routes:

| Route | Training target | Inference cue |
|---|---|---|
| Direct-Control-SFT | final answer only | `<FINAL>` |
| Short-CoT-SFT | short rationale then final answer | `<THINK>` |
| Hybrid-Control-SFT | paired direct and CoT records | either cue |

Fixed for all three:

- Lesson 12 `Math-Physics-CPT` parent;
- 6 layers, 6 heads, width 384, vocabulary 264, context 256;
- 1,000 optimizer updates;
- 4,096 sampled tokens per update;
- optimizer, learning-rate schedule, seed, and checkpoint selection;
- frozen 128-prompt Lesson 12 gate;
- greedy decoding;
- 48 generated tokens for the primary result.

The different targets contain different numbers of assistant tokens. This is
compute/token fairness, not equal-demonstration fairness. The manifest records
eligible targets and approximate target epochs so the distinction remains
auditable.

Secondary diagnostics:

- repeat the same checkpoints at 96 generated tokens;
- record missing `<FINAL>`, missing `<EOS>`, and budget truncation separately;
- score only text after `<FINAL>` in reasoning mode;
- evaluate Hybrid-Control-SFT once per cue;
- report cue compliance, reasoning length, final-answer length, and latency.

The 96-token result is a truncation diagnostic, not the primary score.

## Context extension

The current model uses learned absolute position embeddings. Merely changing
`block_size` does not create meaningful rows 256--511. The conservative pilot
therefore:

1. copies every 256-token CPT tensor exactly;
2. copies learned position rows 0--255 exactly;
3. initializes rows 256--511 by repeating the learned rows 0--255, preserving
   the old prefix exactly while making the positional alias explicit;
4. continues CPT on 512-token windows;
5. optionally trains the hybrid SFT corpus at 512.

This changes the parameter count by only:

```text
(512 - 256) * 384 = 98,304 parameters
```

It does not solve every long-context bottleneck:

| Mechanism | Solves | Does not solve |
|---|---|---|
| learned table extension | valid positions through 512 after training | quadratic training attention |
| ALiBi | position extrapolation | KV-cache size |
| RoPE + YaRN | efficient RoPE context extension | KV-cache size |
| GQA / MQA / MLA | KV-cache and decode bandwidth | position validity |
| sliding-window attention | bounded attention compute/cache | unrestricted access to all old tokens |
| FlashAttention | GPU IO/materialization cost | Nspire CPU kernels or KV size |

The existing incremental C runtime stores FP32 K and V tensors, so its dominant
context-dependent arena term is:

```text
2 * layers * context * width * sizeof(float)
```

For the 6x384 student this is 4.50 MiB at 256 and 9.00 MiB at 512. The current
NGM loader also has a legacy `block_size <= 128` guard; Lesson 14 does not claim
deployment until that format/runtime gate is deliberately revised and aligned.

## Stop rules

- Any frozen-evaluation family in training aborts the build.
- Any rationale whose final answer disagrees with exact ground truth is
  rejected.
- Any record exceeding its declared context is rejected, never silently
  truncated.
- A CoT completion without `<FINAL>` is not allowed to receive answer credit.
- A 512 model is not described as Nspire-ready without export, Host C
  alignment, memory measurement, and physical-device evidence.
