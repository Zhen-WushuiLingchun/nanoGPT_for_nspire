# ADR-0001: Use 2-group GQA and compare learned positions with ALiBi

## Status

Accepted for a PyTorch prototype. C export and Nspire deployment remain gated
on quality, numerical-alignment, and memory evidence.

## Context

Lesson 14 extended the 10.8M byte-level student from context 256 to 512 without
changing multi-head attention. This made a 256-token output budget feasible for
every frozen evaluation prompt, but the current C layout would store 9.00 MiB
of FP32 K/V at context 512. Increasing the output allowance is useful only if
the prompt plus completion fits the model context and the device can retain the
history.

The project needs an architecture experiment that:

- leaves every Lesson 14 MHA checkpoint intact;
- supports context 512 and at least 256 generated tokens;
- materially reduces incremental K/V memory;
- stays simple enough for a scalar C/Ndless implementation;
- separates the attention-head change from the position-encoding change;
- can be initialized from the existing 512-token MHA checkpoint rather than
  starting a new base model from random weights.

## Decision

Build one long-context model with 6 query heads and 2 K/V heads. Each K/V head
serves a group of 3 query heads. Convert the existing MHA checkpoint by copying
Q and averaging K/V projection rows within each group, following the
checkpoint-uptraining idea evaluated by the GQA paper.

Train two variants under the same continuation budget:

1. `GQA-Learned-Context512`: retain the learned 512-row position table. This
   isolates the effect of changing MHA to GQA.
2. `GQA-ALiBi-Context512`: remove the learned position table and add a fixed
   per-query-head linear attention bias. This measures the additional effect of
   an extrapolatable position mechanism.

Both retain 6 layers, width 384, head dimension 64, tanh GELU, tied embeddings,
no linear biases, context 512, the 264-token vocabulary, and the same
250-update CPT plus 1,000-update Hybrid SFT contracts used by the Lesson 14
512 baseline.

The primary exact-answer generation budget becomes 256 tokens for both Direct
and Think. A 384-token run is diagnostic because the longest frozen prompt
leaves only 279 free positions in a 512-token window.

## Consequences

### Positive

- FP32 K/V at context 512 falls from 9.00 MiB to 3.00 MiB.
- The GQA-Learned comparison changes only the K/V head grouping.
- The ALiBi comparison no longer needs a learned position table.
- Fused Q/K/V projection and fixed slopes remain implementable in portable C.
- Existing MHA files, routes, exporter, and runtime remain untouched.

### Negative

- Averaging three K/V heads loses information and may reduce quality.
- ALiBi removes learned absolute-position behavior and may need more
  continuation than 250 updates.
- PyTorch quality does not prove C speed; grouped broadcasting can be cheap in
  storage but still requires six query-head score calculations.
- Supporting the new tensor shapes requires a future NGM version and C kernel.

### Neutral

- Fewer attention parameters make raw parameter-count comparisons unequal.
  The primary fairness contract is fixed width, depth, training tokens, data,
  and context; parameter count and file size are reported as outcomes.

## Alternatives Considered

**MQA with one K/V head**

- Would reduce 512-token FP32 K/V to 1.50 MiB.
- Rejected for the first quality-oriented prototype because it compresses all
  six heads into one K/V representation.

**GQA plus RoPE/YaRN**

- Provides a mature path to context lengths far beyond 512.
- Deferred because rotation, scaling, frequency tables, and C parity add more
  variables than needed for the first device-oriented ablation.

**Sliding-window attention**

- Bounds cache independently of conversation length.
- Deferred because tokens outside the window cease to be fully accessible and
  the experiment would no longer test a full 512-token context.

**MLA**

- Can compress K/V more aggressively.
- Deferred because its projection/decompression path is substantially more
  complex than GQA for the current scalar C teaching runtime.

## References

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
- [DeepSeek-V2 and Multi-head Latent Attention](https://arxiv.org/abs/2405.04434)
