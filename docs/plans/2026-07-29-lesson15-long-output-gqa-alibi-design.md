# Lesson 15: Long-output evaluation and GQA/ALiBi architecture design

## Research questions

1. Does increasing the generation allowance from 96 to 256 tokens improve
   exact math/physics accuracy once context truncation is removed?
2. Can 2-group GQA retain the 512-token MHA baseline quality while reducing
   FP32 K/V memory by 3x?
3. Does replacing the learned position table with ALiBi help or hurt prefix and
   extended-position language-model loss after the same bounded continuation?

## Frozen evidence layers

### Long-output baseline

- checkpoint: `Hybrid-Control-SFT-Context512`;
- frozen Lesson 12 evaluation: 32 prompts per task, 128 total;
- greedy generation;
- Direct and Think both use a 256-token primary allowance;
- Think also uses a 384-token diagnostic;
- exact scoring reads only the final segment;
- budget and context truncation remain separate.

Every selected prompt consumes 20--233 input tokens. Context 512 therefore
leaves at least 279 output positions, so all 256-token primary runs have equal
usable capacity.

### Architecture comparison

| Variable | MHA baseline | GQA-Learned | GQA-ALiBi |
|---|---:|---:|---:|
| query heads | 6 | 6 | 6 |
| K/V heads | 6 | 2 | 2 |
| head dimension | 64 | 64 | 64 |
| learned position rows | 512 | 512 | 0 |
| context | 512 | 512 | 512 |
| CPT updates | 250 | 250 | 250 |
| Hybrid SFT updates | 1,000 | 1,000 | 1,000 |
| sampled tokens/update | 4,096 | 4,096 | 4,096 |

The same `Math-Physics-CPT-Context512` checkpoint initializes both GQA
variants. Q is copied. K and V are reshaped as six 64-dimensional heads, then
the three heads in each group are averaged into two K/V heads. All compatible
embedding, normalization, output-projection, MLP, and tied-head tensors are
copied exactly.

## Metrics

- exact task accuracy at 256 generated tokens;
- mode compliance, `<FINAL>` transition, EOS, budget/context truncation;
- mean reasoning/final tokens and unique final answers;
- validation/test loss;
- validation loss at relative positions 0--255 and 256--511;
- model parameters and raw FP32/W4 weight estimates;
- FP32 K/V bytes at context 256 and 512;
- CUDA training throughput and peak allocation;
- initialization tensor-copy/averaging invariants.

Loss is compared only between checkpoints evaluated on the same corpus and
target format. Direct and Think token losses are not treated as interchangeable
ability metrics.

## Gates

- Existing MHA checkpoints and loaders must still pass every test.
- MHA-to-GQA conversion must be deterministic and hash-checked.
- All non-attention tensors must be exact copies, except the ALiBi variant's
  deliberate removal of the position table.
- GQA K/V memory must equal one third of MHA for the same context.
- No C/Nspire-ready claim before a new format, Host C logits alignment, and
  physical-device measurement.

## Literature boundary

The GQA paper reports checkpoint uptraining with a small fraction of original
pretraining compute and quality near MHA at its scale. ALiBi reports
train-short/test-long behavior. These motivate the experiment; they do not
guarantee success for a 10.8M byte-level model or a 250-step continuation.
