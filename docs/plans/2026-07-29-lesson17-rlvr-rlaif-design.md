# Lesson 17 design: RLVR, direct-RLAIF, and their combination

## Status

Approved by the user on 2026-07-29. This design converts the four-route matrix
frozen before Lesson 16 into an executable reinforcement-learning experiment.

## Research questions

1. Can group-relative policy optimization with an exact local verifier improve
   held-out arithmetic or numeric-physics accuracy in a roughly 9.5M-parameter
   byte-level policy?
2. Can direct AI feedback improve prompt-value use, explanation coherence, and
   reasoning/final consistency without improving exact answers?
3. Does combining exact reward with bounded AI feedback outperform either
   reward source alone?
4. Does RL improve Think more than Direct under the same 256-token output cap?
5. Is the Lesson 15 parent or Lesson 16 SFT v2 checkpoint the more exploitable
   RL starting policy?

No improvement is assumed. Sparse or zero-variance reward, reward hacking,
format collapse, and challenge-set regression are first-class results.

## Why direct-RLAIF

Three possible AI-feedback routes were considered:

1. Train a reward model on DeepSeek preferences, then run RL.
2. Use DeepSeek directly as the reward provider during RL.
3. Use DeepSeek preferences with DPO.

Route 2 is selected. The RLAIF study explicitly names this direct-RLAIF
(`d-RLAIF`): an off-the-shelf LLM directly provides rewards during RL, avoiding
reward-model staleness and a separate reward-model training stage. DPO remains a
useful later preference-optimization control, but it is not called RL here.

Primary references:

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [RLAIF and direct-RLAIF](https://arxiv.org/abs/2309.00267)
- [DeepSeek V4 API models](https://api-docs.deepseek.com/quick_start/pricing/)
- [DeepSeek V4 thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)

These papers use far larger policies and much larger compute. They motivate the
experiment and terminology; they do not predict success for this project.

## Frozen route matrix

| Route | Policy update | Exact verifier reward | DeepSeek reward |
|---|---:|---:|---:|
| SFT-only | no | no | no |
| RLVR | group-relative RL | yes | no |
| direct-RLAIF | group-relative RL | evaluation only | yes |
| RLVR + direct-RLAIF | group-relative RL | yes | bounded auxiliary |

The three trainable routes use the same starting checkpoint, prompt schedule,
mode schedule, sampling temperature, group size, output-token cap, optimizer
updates, fixed reference policy, KL coefficient, clipping coefficient, and
policy seeds.

## Stage 0: objective dual-start screen

The Lesson 15 parent has better 256-family challenge accuracy (`3/256` versus
`1/256`), while SFT v2 has perfect Think termination. Neither is silently
chosen.

Both checkpoints sample the same 32 training prompts, modes, RNG seeds, group
size, temperature, and 256-token cap with no parameter update. Selection order:

1. higher fraction of groups containing both at least one exact-correct and one
   incorrect completion;
2. higher exact-correct sampled-completion rate;
3. lower invalid-format rate;
4. exact tie falls to SFT v2 because it has the cleaner termination contract.

The first metric measures whether group-relative exact reward has usable
within-group variance. The screen is training-only and never reads primary or
challenge evaluation outcomes beyond the already published Lesson 16 record.

## Rollout curriculum

Build a deterministic English-only training pool containing:

- small integer and signed arithmetic;
- decimal arithmetic and exact integer division;
- introductory numeric physics across all ten existing formula families;
- balanced Direct and Think control modes.

Every answer is recomputed locally. Families from the Lesson 12 primary
evaluation and Lesson 16 challenge evaluation are excluded before sampling.
The policy never updates on either holdout.

Formal schedule per route and policy seed:

| field | value |
|---|---:|
| policy seeds | 20260731, 20260732, 20260733 |
| rollout batches | 16 |
| optimizer updates | 32 (2 per rollout batch) |
| prompt groups/update | 4 |
| candidates/group | 8 |
| sampled completions/route/seed | 512 |
| output cap | 256 tokens |
| modes/update | 2 Direct + 2 Think |
| sampling temperature | 0.8 |
| policy epochs/rollout batch | 2 |
| policy microbatch | 4 trajectories, gradient-accumulated per epoch |
| optimizer | AdamW |
| learning rate | `5e-6` |
| GRPO clip epsilon | 0.2 |
| fixed-reference KL beta | 0.02 |
| max gradient norm | 1.0 |

All routes use the final update; holdout metrics do not select a checkpoint.

Before any RL result was produced, implementation exposed an ambiguity in the
original table: 16 batches and two genuine policy epochs cannot also mean only
16 optimizer steps. Lesson 17 therefore freezes 16 rollout batches and one
optimizer step after each of the two policy epochs, for 32 steps total. Each
epoch covers all 32 trajectories with four-trajectory gradient-accumulation
microbatches; the accumulated loss is weighted by generated-token count. This
clarification was made before the start screen or any policy training, and is
identical across all three trainable routes.

## Policy objective

For each prompt group, normalize scalar rewards:

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-6)
```

Generated-token log probabilities use the same temperature-scaled policy
distribution as sampling. The clipped group-relative surrogate is:

```text
ratio_t = exp(logp_current_t - logp_old_t)
policy_t = min(ratio_t * A, clip(ratio_t, 0.8, 1.2) * A)
```

The fixed starting checkpoint supplies reference log probabilities. The
per-token non-negative sampled KL estimator is:

```text
delta = logp_reference - logp_current
KL_sample = exp(delta) - delta - 1
```

Only generated tokens participate. Prompt tokens, right-padding, and tokens
after a terminal event are masked. A zero-variance group has zero policy
advantage and contributes only the fixed-reference KL term.

This is a small GRPO-style teaching implementation, not a claim of reproducing
DeepSeekMath's distributed training system.

## Deterministic verifier reward

The existing strict final-answer scorer supplies components:

```text
0.80 * numeric_correct
+ 0.15 * (numeric_correct and unit_correct)
+ 0.05 * format_valid
```

Properties:

- fully correct numeric/unit/format output scores `1.0`;
- a wrong numeric output scores at most `0.05`;
- an exact numeric answer cannot be outweighed by style;
- no Python `eval` or model judge participates;
- reward components and final scalar are recorded separately.

## DeepSeek direct-AI reward

Current live `/models` verification on 2026-07-29 confirms
`deepseek-v4-pro`. Each request contains one training prompt and a deterministically
shuffled candidate group. It never contains the locally verified answer.

The versioned rubric scores each candidate from 0 to 4 on:

- uses quantities actually present in the prompt;
- mathematically or physically coherent explanation;
- reasoning and final answer are mutually consistent;
- avoids unsupported claims;
- stays concise under the device-oriented budget.

The response is strict JSON with all candidate IDs, integer scores, a preferred
candidate ID, and a short public rationale. Hidden provider reasoning is never
stored. Scores are divided by four to obtain `[0,1]`. Invalid model format
forces the AI reward to zero, but local numeric correctness is not part of the
RLAIF-only reward.

The public output budget is 4096 provider tokens. A pre-training live probe
with 1024 tokens produced empty public content on all three bounded retries
because high thinking consumed the available output budget. No score was
accepted or cached. The budget was raised before any AI-rewarded policy update;
model, rubric, reward mapping, and all policy hyperparameters remained fixed.

Live V4 responses also exposed one semantically equivalent JSON variation:
some valid judgments encode `scores` as an eight-entry
`candidate_id -> integer score` object instead of the requested list of
objects. The parser accepts only those two exact containers, requires the
candidate-ID set to match exactly, enforces integer scores in `[0,4]`, checks
that the preferred ID has a maximum score, and normalizes the public cache to a
sorted list. Empty, missing, partial, extra-ID, fractional, or inconsistent
score containers remain hard failures. This compatibility was fixed from
provider behavior rather than inferred scores.

The provider request uses short deterministic aliases `C0` through `C7` in the
already shuffled candidate order. A private in-memory map connects each alias
to the full trajectory ID. The response must contain the complete alias set
exactly once before scores are mapped back to trajectory IDs. This removes
observed long-ID copying errors without using position as a fallback or
guessing a missing identity. Changing the alias protocol changes the canonical
request hash, so pre-alias cache entries cannot be silently reused.

Every record stores model ID, rubric version, request-body SHA-256, provider
request ID, token usage, candidate permutation, retry count, and parsed public
feedback. Credentials are read only at HTTP call time and are prohibited from
all serializable configuration and output.

## Combined reward and anti-hacking order

```text
combined = verifier_reward + 0.20 * ai_reward
```

A numerically wrong but AI-preferred response can score at most `0.25`.
A numerically correct response scores at least `0.95`, even without EOS.
Therefore DeepSeek can break ties among similarly correct or similarly wrong
responses, but cannot override the local exact checker.

Record and audit:

- high AI score plus wrong numeric result;
- low AI score plus correct numeric result;
- reward/held-out-accuracy correlation;
- zero-variance group rate;
- group reward standard deviation;
- KL to the starting policy;
- response length and common-final concentration;
- provider score position bias by shuffled candidate slot.

## Evaluation and claim gates

Every final policy is greedily evaluated with the same 256-token cap on:

1. Lesson 12 primary 128-family set;
2. Lesson 16 challenge 256-family set;
3. Direct and Think separately.

Report each policy seed, mean, standard deviation, and all raw counts. An RL
route is called an ability improvement only if:

- mean exact count exceeds SFT-only on both primary and challenge sets;
- at least two of three seeds improve rather than one lucky seed;
- format/mode compliance is at least 95%;
- no holdout family was used for rollout or reward;
- high reward is not explained only by length, EOS, or judge preference.

Failure to meet these gates is retained as a negative result. No post-hoc
threshold change is allowed.

## Scope boundary

Lesson 17 ends at PyTorch policy optimization and frozen Host GPU evaluation.
It does not include quantization, C kernels, NGM export, PyTorch/C alignment,
Ndless packaging, or physical-device timing. Those remain downstream only
after a policy route demonstrates a reproducible ability benefit.
