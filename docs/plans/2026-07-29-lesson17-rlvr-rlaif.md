# Lesson 17 implementation plan: RLVR and direct-RLAIF

> Execute in testable batches. Provider calls begin only after the request
> planner, strict response parser, credential scan, and local fake transport
> tests pass.

## Task 1: Freeze the RL curriculum and verifier

**Output**

- `training/nanogpt_nspire/lesson17_data.py`
- `training/nanogpt_nspire/rl_rewards.py`
- `tests/python/test_lesson17_data.py`
- `tests/python/test_rl_rewards.py`

**Contract**

- deterministic, family-disjoint numeric training prompts;
- explicit Direct/Think schedule;
- no primary/challenge family overlap;
- exact numeric/unit/format reward components;
- wrong numeric output cannot score above `0.05`;
- combined AI auxiliary reward cannot outrank a correct numeric completion.

**Test**

```powershell
python -m pytest tests/python/test_lesson17_data.py `
  tests/python/test_rl_rewards.py -q
```

## Task 2: Implement stochastic grouped rollouts

**Output**

- `training/nanogpt_nspire/rl_rollout.py`
- `tests/python/test_rl_rollout.py`

**Contract**

- batch eight candidates for one prompt;
- sample from the temperature-scaled full vocabulary;
- record every generated token and old-policy log probability;
- include the sampled EOS or leaked control token in the policy trajectory;
- decode Direct/Think segments with the existing grammar;
- distinguish EOS, budget, context, and special-token termination;
- deterministic under a frozen CPU sampling generator.

**Test**

```powershell
python -m pytest tests/python/test_rl_rollout.py -q
```

## Task 3: Implement GRPO-style policy updates

**Output**

- `training/nanogpt_nspire/group_policy_train.py`
- `tests/python/test_group_policy_train.py`

**Contract**

- per-group normalized advantages;
- generated-token-only clipped ratio objective;
- fixed-reference sampled KL;
- right-padding excluded by a loss mask;
- two genuine policy epochs and optimizer steps per rollout batch;
- four-trajectory microbatches accumulated within each policy epoch;
- finite loss/gradient checks;
- strict parent route/hash/model architecture;
- checkpoint and trajectory summaries remain credential-free.

**Test**

```powershell
python -m pytest tests/python/test_group_policy_train.py -q
```

## Task 4: Implement credential-isolated direct-RLAIF

**Output**

- `training/nanogpt_nspire/preference_judge.py`
- `tests/python/test_preference_judge.py`

**Contract**

- `deepseek-v4-pro`, OpenAI Chat Completions, thinking high, JSON output;
- deterministic candidate shuffle and versioned rubric;
- strict one-score-per-candidate parser;
- score bounds, preferred-ID consistency, response byte cap, retries;
- request hash and public provenance without request headers or key;
- live key read only inside transport;
- concurrent request budget remains exact.

**Test**

```powershell
python -m pytest tests/python/test_preference_judge.py -q
```

## Task 5: Run the dual-start screen

**Output**

- ignored `artifacts/lesson17-start-screen/`

**Contract**

- Lesson 15 parent and Lesson 16 SFT v2;
- identical 32 prompts, modes, temperature, group size, and sampling seeds;
- no updates and no API calls;
- select by the frozen mixed-exact-group/exact-rate/format order.

## Task 6: Run one local end-to-end smoke

**Output**

- one-update RLVR checkpoint and run record;
- fake-judge RLAIF and combined smoke records.

**Gate**

- finite gradients and checkpoint;
- old/current/reference log-prob shapes align;
- reward and KL histories serialize;
- strict evaluator loads the new route;
- no holdout or credential in trajectories.

## Task 7: Collect live direct-AI rewards and train all routes

**Output**

- ignored provider cache keyed by request SHA-256;
- RLVR, direct-RLAIF, and combined checkpoints for three seeds;
- SFT-only frozen reference record.

**Contract**

- 16 rollout batches and 32 optimizer steps, four groups/batch, eight
  candidates/group;
- 256-token rollout cap, temperature 0.8;
- two policy epochs, LR `5e-6`, clip `0.2`, KL beta `0.02`;
- exact same prompt/mode schedule per seed and route;
- API failures remain explicit; no fabricated preference;
- cached valid responses are reused only for an identical request hash.

## Task 8: Evaluate and diagnose

**Output**

- primary/challenge Direct/Think JSON for every final policy;
- per-seed and aggregate experiment record;
- high-reward-wrong and low-reward-correct audit.

**Gate**

- no training family intersects holdout;
- all result hashes and counts recorded;
- claims require both-set mean improvement and two-of-three seed support;
- otherwise publish the negative result without changing thresholds.

## Task 9: Document, verify, and publish

**Output**

- `docs/lessons/17-rlvr-rlaif-and-reward-hacking.md`
- `experiments/lesson17-rlvr-rlaif.json`
- README update.

**Verification**

```powershell
python -m pytest -q
cmake -S . -B build/host -G "Visual Studio 17 2022" -A x64
cmake --build build/host --config Release
ctest --test-dir build/host -C Release --output-on-failure
python -m pytest -q
git diff --check
```

Scan tracked and generated public artifacts for credential-shaped data.
Checkpoints, provider caches, trajectories, raw rollout data, and build outputs
remain ignored.
