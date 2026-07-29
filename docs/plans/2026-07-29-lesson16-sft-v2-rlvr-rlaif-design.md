# Lesson 16 design: SFT v2 and frozen RLVR/RLAIF protocol

## Status

Design accepted in principle. This document freezes the questions and evidence
layers before generating new teacher feedback or updating a policy.

## Objective

Create a stronger and cleaner post-training starting point for the 512-context
student, then pre-register a four-route comparison that separates deterministic
verifiable reward from DeepSeek AI preference.

Lesson 16 prepares and evaluates SFT v2. Lesson 17 executes the RL routes.

## SFT v2 questions

1. Can explicit short-process targets raise Think format/EOS completion above
   95% under a 256-token allowance?
2. Can balanced number, sign, unit, and formula coverage reduce common-number
   attractors on frozen and range-shifted prompts?
3. Do compact, independently verified positive targets improve consistency
   between prompt values, substitution, intermediate arithmetic, and final
   answer on held-out hard-negative probes?
4. Does Think improve exact accuracy over Direct in the same checkpoint under
   the same output-token cap?

## SFT v2 data contract

- English only.
- Existing 264-token vocabulary and `<THINK>` / `<FINAL>` controls.
- One to four compact reasoning steps.
- Exact locally generated arithmetic and introductory numeric physics.
- Public worked examples only when their license and split lineage are known.
- Frozen family-level exclusion for every existing evaluation family.
- Balanced operators, signs, decimals, magnitude buckets, formula families,
  and unit spellings.
- Explicit hard negatives are stored as adversarial evaluation and future
  preference/RL examples, never as positive next-token targets. Pure SFT does
  not learn from records that never participate in its loss.
- Every numeric target independently recomputed by a non-`eval` parser.
- DeepSeek-generated visible text is labeled synthetic hard-target data, not
  logit distillation.

## Frozen 256-token evaluation

The primary comparison uses greedy decoding and the same prompt set for Direct
and Think. Report:

- exact value and unit accuracy;
- format, mode, `<FINAL>`, EOS, role leakage, repetition;
- budget and context truncation separately;
- mean and percentile reasoning length;
- exact accuracy per generated token;
- common-final concentration;
- in-range, range-shifted, sign-shifted, and substitution-adversarial slices.

SFT v2 must improve format/EOS without reducing exact holdout accuracy before
it becomes the RL starting checkpoint.

## Lesson 17 route matrix

| Route | Exact verifier | DeepSeek preference |
|---|---:|---:|
| SFT-only | no update | no update |
| SFT + RLVR | reward | none |
| SFT + RLAIF | evaluation guard only | reward |
| SFT + RLVR + RLAIF | reward | reward |

Freeze the same starting checkpoint, rollout prompts, rollout-token budget,
optimizer-update budget, sampling temperature, KL/reference-policy contract,
and random seeds across trainable routes.

## Reward boundaries

### Deterministic RLVR reward

- parsed numeric correctness;
- unit equivalence;
- required control-token transition and EOS;
- no special-token or role leakage;
- bounded repetition and length;
- optional symbolic formula equivalence where the local checker supports it.

### DeepSeek AI preference

DeepSeek compares candidate responses using a versioned rubric:

- uses the values actually present in the prompt;
- explanation is physically and mathematically coherent;
- reasoning and final answer are mutually consistent;
- no unsupported conceptual claim;
- concise enough for the device budget;
- useful treatment of non-numeric conceptual questions.

The judge does not override a deterministic wrong-answer result. API responses,
rubric version, model identifier, sampling settings, request hashes, and parsed
preference labels are recorded without credentials.

## Anti-reward-hacking gates

- independent verifier implementation for the final holdout;
- reward holdout prompts never sampled for policy updates;
- report reward/accuracy correlation and disagreements;
- audit high-reward wrong answers and low-reward correct answers;
- cap format and style reward so they cannot outweigh a wrong numeric answer;
- compare against a randomized/preference-label control if budget permits;
- at least three policy seeds before claiming an RL improvement.

## Literature boundary

DeepSeek-R1 motivates cold-start SFT followed by RL and documents undesirable
behaviors in its pure-RL R1-Zero route. DeepSeekMath motivates GRPO for
mathematical reasoning. Constitutional AI defines an RLAIF pipeline based on
AI preferences, rather than teacher answer generation. Process-supervision
work motivates checking intermediate reasoning. None of those results is
assumed to transfer automatically to a roughly 10M-parameter byte model.

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath and GRPO](https://arxiv.org/abs/2402.03300)
- [Constitutional AI and RLAIF](https://arxiv.org/abs/2212.08073)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
