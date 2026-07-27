# Lesson 07 Teacher v2 and Conditional Distillation Design

## 1. Decision

Teacher v2 is a single-variable generalization experiment:

```text
Teacher v1 dropout = 0.2
Teacher v2 dropout = 0.3
```

Every other architecture, initialization, data, optimizer, schedule,
evaluation, checkpoint-selection and training-budget field remains identical.
This makes the result an interpretable dropout ablation rather than an
uncontrolled hyperparameter bundle.

## 2. Evidence motivating v2

Teacher v1 used 10,695,936 parameters and reached its best fixed-window
validation loss at step 2250:

```text
step 2250 loss = 1.4839615273475646
```

It then stayed near `1.484–1.490` through step 3750 before degrading
continuously to:

```text
step 10000 loss = 1.7740670037269592
```

Over the same interval, sampled training loss continued falling. The widening
train/validation gap is evidence for overfitting. Best-checkpoint selection
prevented the final model from using the step-10000 weights, but it did not
make the best checkpoint pass the preregistered source-quality gate.

## 3. Alternatives not selected

### Stronger dropout plus stronger weight decay and early stopping

This may have a higher chance of improving the result, but three simultaneous
changes would prevent attribution. Early stopping also reduces wasted compute
without necessarily improving the already selected best checkpoint.

### Smaller Teacher

A narrower or shallower model may overfit less, but it would change the
capacity premise of Quantized-Small and no longer isolate training
regularization.

### Multiple-candidate search

Trying several dropout values and reporting the best would tune against the
same validation set used by the final comparison. This lesson permits one
preregistered v2 attempt only.

## 4. Frozen Teacher v2 protocol

```text
route                 Teacher-v2
vocabulary            dataset vocabulary, expected 65
context               128
layers                6
heads                 6
embedding width       384
MLP ratio             4
dropout               0.3
bias                  false
tied embedding        true
seed                   1337
steps                  10,000
batch size             64
training tokens        81,920,000
maximum learning rate  0.001
minimum learning rate  0.0001
warmup steps           100
weight decay           0.1
AdamW betas            0.9, 0.99
gradient norm cap      1.0
validation interval    250
validation batches     50
validation seed        1338
sample seed            1340
sample temperature     0.8
```

The unchanged gate is:

```text
selected validation loss <= 1.4797899746894836
```

Teacher v1 remains immutable. Teacher v2 writes a separate checkpoint and run
directory.

## 5. Reproducibility invariants

With the same construction seed, v1 and v2 must begin with bit-identical
parameters. Dropout probability is not a Parameter, so it must not alter model
initialization.

Tests require the complete serialized training configurations to differ only
in:

```text
dropout
output_dir
```

Run identity also differs because v2 has its own route, checkpoint filename
and deployment interpretation.

The source implementation commit is frozen before training. The selected
checkpoint is independently reloaded and checked for:

- exact parameter count;
- tied token/head identity;
- exact fixed-window loss reproduction;
- exact fixed-seed sample reproduction;
- causal future isolation;
- artifact byte count and SHA-256;
- unchanged quality threshold.

## 6. Stop rule

If Teacher v2 fails the unchanged gate:

- preserve and document the failed run;
- do not train Distilled-Small;
- do not replace Teacher v1 evidence;
- do not try dropout 0.25, 0.35 or another optimizer in this lesson;
- record that the one-shot regularization hypothesis did not unlock the route.

This prevents repeated validation-set tuning.

## 7. Conditional Distilled-Small design

Only a passing Teacher v2 unlocks distillation.

The student is exactly Direct-Small v1:

```text
layers            4
heads             5
embedding width   160
context           128
dropout           0.1
bias              false
tied embedding    true
parameters        1,261,120
```

It reuses the Direct-Small initialization seed, batch sequence, optimizer,
learning-rate schedule, 5,000 updates and 40,960,000 student training tokens.
The training objective is the only intended primary difference.

For student logits `z_s`, Teacher logits `z_t`, hard labels `y`,
temperature `T=2.0` and soft-loss weight `alpha=0.5`:

```text
hard_loss = cross_entropy(z_s, y)

soft_loss = T^2 * KL(
    softmax(z_t / T)
    ||
    softmax(z_s / T)
)

total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
```

The Teacher runs in eval mode under inference/no-grad and is never updated.
Multiplication by `T^2` preserves useful gradient scale when temperature
softens the distributions.

Distilled-Small succeeds as a measured comparison only if its fixed-window
selected validation loss is lower than Direct-Small's:

```text
1.4997899746894836
```

Regardless of outcome, the run reports hard loss, soft loss, total loss,
Teacher inference cost, student training time, checkpoint size and the same
deployment-pending boundaries as Direct-Small.

## 8. Route boundaries

Passing Teacher v2 would unlock two later uses:

1. a formal packed INT4 rerun sourced from v2;
2. Distilled-Small training in this lesson.

The existing v1 INT4 artifact remains diagnostic. A passing PyTorch Teacher or
student still does not establish Host C, integer-kernel or Nspire inference
success.
