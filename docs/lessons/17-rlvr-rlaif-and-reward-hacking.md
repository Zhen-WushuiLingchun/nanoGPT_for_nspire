# Lesson 17：RLVR、direct-RLAIF 与奖励稀疏/奖励错位

Lesson 16 的 SFT v2 把 Think 终止率修到 100%，却没有提升冻结数值正确率；
256-family challenge 甚至从父模型的 `3/256` 降到 `1/256`。这说明继续堆
correct target 的交叉熵，未必能让 9.5M 参数 byte-level GPT 学会可泛化算法。

本课第一次让模型自己采样多个答案，再根据结果得到 reward 并更新 policy：

| 路线 | 本地 exact verifier | DeepSeek AI feedback | 主要问题 |
|---|---:|---:|---|
| SFT-only | 无更新 | 无更新 | 冻结基线 |
| RLVR | 主奖励 | 无 | 可验证正确性是否能产生能力提升？ |
| direct-RLAIF | 只做审计，不进 reward | 主奖励 | AI 评价能否改善回答质量？ |
| RLVR + direct-RLAIF | 主奖励 | 0.20 上限辅助 | 两种信号能否互补？ |

机器可读结果和预注册合同位于：

- [`lesson17-rlvr-rlaif.json`](../../experiments/lesson17-rlvr-rlaif.json)
- [`Lesson 17 design`](../plans/2026-07-29-lesson17-rlvr-rlaif-design.md)
- [`Lesson 17 implementation plan`](../plans/2026-07-29-lesson17-rlvr-rlaif.md)

本课只覆盖 PyTorch policy optimization 和 Host GPU 冻结评测。没有量化、
C kernel、`.ngm` 导出、PyTorch/C 对齐或 Nspire 真机结论。

## 1. 这次哪些东西才叫 RL

一次 SFT update 已经知道标准 target：

```text
prompt -> fixed correct assistant tokens -> cross-entropy
```

一次本课的 policy update 则是：

```text
prompt
  -> current policy samples 8 different completions
  -> verifier or AI judge assigns scalar rewards
  -> compare each completion with its same-prompt peers
  -> increase/decrease sampled-token probability
```

模型生成的 token 影响 reward，reward 又反过来影响生成概率，因此这是 on-policy
reinforcement learning 教学实现。它采用 GRPO-style group-relative advantage
和 PPO-style clipped ratio，但不声称复现 DeepSeekMath 的大规模分布式系统。

### 1.1 RLVR

RLVR 是 reinforcement learning with verifiable rewards。数值题的最终值和单位
可以由确定性程序检查，因此 reward 不依赖另一个模型的“感觉”：

```text
0.80 * numeric_correct
+ 0.15 * (numeric_correct and unit_correct)
+ 0.05 * format_valid
```

错误数值最多得到 `0.05`；正确数值与单位至少得到 `0.95`。解析器沿用冻结
`Decimal`/unit/termination 检查，不使用 Python `eval`。

### 1.2 direct-RLAIF

RLAIF 是 reinforcement learning from AI feedback。本课选择 direct-RLAIF：
DeepSeek V4-Pro 在 policy training 时直接给候选 reward，不先训练 reward
model。这样避免了额外 reward-model stage 和 reward-model staleness，但代价是：

- API 延迟与费用进入训练环；
- provider JSON 可能漂移；
- AI judge 可能偏好流畅但数值错误的答案；
- 同一个 judge 同时扮演“评委”和风格来源，容易出现 reward bias。

这与 Lesson 13 的 API 合成数据不同。合成答案再做 CE 是 synthetic-data SFT；
student 学 teacher logits 才是严格 distillation；本课则让 AI 只评价 policy
自己采样的候选，并把标量分数用于 policy gradient。

主要方法参考：

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [RLAIF 与 direct-RLAIF](https://arxiv.org/abs/2309.00267)

## 2. 为什么先做双起点筛选

Lesson 15 parent 在 challenge 上较好，Lesson 16 v2 的 Think 终止更干净。不能
看完 RL 结果再挑有利起点，因此先对两个 checkpoint 做无更新、无 API 的相同
采样：

- 32 个训练 prompt；
- 每组 8 个候选；
- temperature `0.8`；
- 最多 256 个新 token；
- 完全相同的 prompt/mode/RNG；
- 不读取 primary/challenge 的新结果来选择。

选择顺序预先固定为：

1. 同一组内同时有正确与错误候选的 group 比例；
2. 全部采样 completion 的精确正确率；
3. 更低格式无效率；
4. 完全相同才选 SFT v2。

| 起点 | mixed-exact groups | exact completions | invalid format |
|---|---:|---:|---:|
| Lesson 15 SFT | `6/32` | `6/256` | `0/256` |
| Lesson 16 SFT v2 | `4/32` | `5/256` | `1/256` |

因此三条 RL 路线共同使用 Lesson 15：

```text
f385a51f7ac4d53b1a640f9a977308a365bb032a6a2dbe53394ba035169513ee
```

这里 mixed-exact 比单纯 accuracy 更重要。group-relative advantage 需要同组
reward 有方差；如果 8 个候选全错或全对，这一组的 normalized advantage 为
零，只剩 KL 约束。

## 3. 训练数据为什么不能碰 holdout

Lesson 17 重新确定性生成 64 个训练 family：

- 32 个整数、负数、小数、整除和括号算术；
- 32 个覆盖既有十类公式的 numeric physics；
- 每个 update 固定 2 个 arithmetic、2 个 physics；
- 每个 update 固定 2 个 Direct、2 个 Think。

构建前排除 Lesson 12 primary 与 Lesson 16 challenge 的全部 `1041` 个 family。
最终报告重新读取训练 `problem_pool.jsonl` 和两份 holdout，集合求交必须为零；
仅仅“代码中说排除了”不算证据。

## 4. 分组 rollout 如何保持 token 对齐

每个 prompt 一次 batch 采样 8 个候选。采样分布是：

```text
p_old(token | prefix) = softmax(logits / 0.8)
```

使用冻结 CPU `torch.Generator` 从 GPU 计算出的概率采样，保证同 seed 可重放。
轨迹必须保存：

- 完整 prompt tokens；
- 每一个生成 token；
- 每一个 token 在旧策略下的 log-prob；
- EOS、第一次 Think `<FINAL>` transition；
- leaked special token；
- budget/context/format 终止原因。

EOS 或非法 special token 也必须留在训练轨迹中。若生成时把 EOS 计入 reward，
训练时却不优化它，behavior policy 与 policy loss 就不再是同一事件。

右 padding 只为 microbatch 张量对齐。loss mask 从
`len(prompt_tokens)-1` 开始，恰好让第一个生成 token 成为 next-token target；
prompt、padding 和终止后的 token 都不参与 policy objective。

## 5. Group-relative advantage 与 clipped objective

同一 prompt 的 8 个 reward 做 population normalization：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-6)
```

零方差组令 `A_i = 0`。对每个生成 token：

```text
ratio_t = exp(logp_current_t - logp_old_t)

surrogate_t = min(
  ratio_t * A_i,
  clip(ratio_t, 0.8, 1.2) * A_i
)
```

固定起点 checkpoint 作为 reference policy。使用非负 sampled KL estimator：

```text
delta_t = logp_reference_t - logp_current_t
KL_t = exp(delta_t) - delta_t - 1

loss = -mean(surrogate_t - 0.02 * KL_t)
```

每个 rollout batch 有 32 条轨迹。为了控制显存，每 4 条做一个 microbatch，
按生成 token 数加权累积；每批做两个真正的 policy epoch 和两个 optimizer
step。最终合同是 16 个 rollout batches、32 个 optimizer steps，而不是把
“16 updates”和“2 epochs”含糊地写成同一个数字。

正式超参数：

| 字段 | 值 |
|---|---:|
| policy seeds | `20260731`, `20260732`, `20260733` |
| candidates/group | 8 |
| groups/rollout batch | 4 |
| completions/route/seed | 512 |
| output cap | 256 |
| policy epochs/batch | 2 |
| learning rate | `5e-6` |
| clip epsilon | `0.2` |
| reference KL beta | `0.02` |
| max gradient norm | `1.0` |

## 6. RLAIF 请求不能携带本地标准答案

每个 DeepSeek 请求只包含：

- 原题；
- `direct` 或 `think` mode；
- 8 个 policy completion；
- 评分 rubric。

它不包含 `expected_answer`、`expected_unit` 或本地 formula。否则 AI judge 只是
换一种外部 verifier，无法回答“无 ground truth 的 AI feedback 有何贡献”。

rubric 要求检查：

- 是否使用题目给出的量；
- 数学/物理是否连贯；
- reasoning 与 final 是否一致；
- 是否有无依据断言；
- 是否简洁，不奖励冗长。

每个候选得 0–4 整数分，再除以 4。RLAIF-only 只用该分数，但本地
`format_valid=false` 会把它归零；本地 numeric correctness 仅用于事后审计。

组合路线为：

```text
combined = verifier + 0.20 * ai_reward
```

所以数值错误但 AI 满分最多 `0.25`，数值正确最少 `0.95`。AI 可以在相似正确
或相似错误答案间打破平局，不能压过 exact correctness。

## 7. Live provider 暴露的三个工程现象

### 7.1 High thinking 吃完公开输出预算

最初 `max_tokens=1024` 的同一请求连续三次 `content=""`。严格 parser 全部
拒绝，没有分数进入 cache 或 policy update。将 provider 输出预算提高到 4096
后公开 JSON 才稳定出现；policy 超参数、rubric 和 reward 不变。

### 7.2 等价 JSON 容器漂移

有时 V4-Pro 返回：

```json
{"scores":[{"candidate_id":"C0","score":4}]}
```

有时返回：

```json
{"scores":{"C0":4}}
```

parser 只接受这两种精确定义，随后要求 8 个 ID 完整、唯一，分数必须为 0–4
整数，preferred 必须属于最大分。项目内部 cache 永远规范化为排序后的列表。

### 7.3 长 candidate ID 会被抄错

完整轨迹 ID 很长：

```text
policy-20260732:update-06:slot-3:candidate-7
```

provider 曾连续三次漏写或改写其中一个 ID。不能按返回位置猜它是谁。最终请求
在确定性 shuffle 后使用短别名 `C0`–`C7`，内存中保留 alias→trajectory map；
只有完整 alias 集合通过后才映回真实 ID。协议改变会改变 request SHA，因此
旧 cache 不会静默复用。

所有缓存都是 append-only JSONL，以 canonical request-body SHA-256 为键。
公开记录只保存 model、request ID、usage、规范化分数、简短 rationale 和 retry
次数；API key 与 provider hidden reasoning 不写入文件。

## 8. 训练信号到底有多稀疏

| 路线 | seed exact / 512 | zero-variance groups / 64 | mean reward 或 AI score |
|---|---|---|---|
| RLVR | `10, 5, 7` | `55, 56, 56` | `0.0685, 0.0590, 0.0627` |
| direct-RLAIF | `9, 6, 10` | `55, 57, 57` | `0.0273, 0.0132, 0.0186` |
| combined | `9, 6, 10` | `54, 56, 55` | `0.0726, 0.0633, 0.0718` |

每条路线每 seed 有 64 组，但大约 54–57 组没有相对 reward 方差。也就是说
绝大多数 prompt 只产生“全错且同分”的候选，真正驱动 policy gradient 的组
不到十组左右。这是小模型 RL 的核心限制，不是多跑几个 epoch 就能消失。

AI score 与 numeric correctness 的训练样本 Pearson correlation 为
`0.756–0.938`，但 AI 平均分极低且非零样本很少。审计还观察到：

- direct-RLAIF 有 2 个 `AI >= 0.75` 但 numeric wrong；
- direct-RLAIF 有 5 个 `AI <= 0.25` 但 numeric correct；
- combined 的 0.20 上限成功阻止这些分歧压过 exact verifier。

这就是为什么 RLAIF 不能被叫做“更聪明的 verifier”：它提供的是不同且可能
错位的偏好信号。

## 9. 冻结 holdout 结果

所有 checkpoint 都用 greedy decoding、Direct/Think 各自 256-token cap：

- Primary：每 mode 128 题，合并 256 次生成；
- Challenge：每 mode 256 题，合并 512 次生成。

| 路线 | Primary exact / 256（三 seeds；mean） | Challenge exact / 512（三 seeds；mean） | 最低 format/mode | 同时改善两套 | ability gate |
|---|---|---|---:|---:|---:|
| SFT-only | `4` | `6` | `87.89%` | 基线 | 基线 |
| RLVR | `6, 8, 4`；`6.00` | `5, 4, 7`；`5.33` | `87.50%` | `0/3` | **FAIL** |
| direct-RLAIF | `0, 6, 3`；`3.00` | `5, 6, 9`；`6.67` | `86.72%` | `0/3` | **FAIL** |
| RLVR + direct-RLAIF | `6, 7, 5`；`6.00` | `5, 4, 7`；`5.33` | `87.50%` | `1/3` | **FAIL** |

表中的 format/mode 是每条 RL 路线在两套合并 Direct+Think 结果中的最小值。
Direct 和 challenge 本身均为 100%；回退来自 primary 的 Think 输出。这里不能
因为 SFT-only 自己只有 `87.89%` 就在看完结果后放宽 95% 的预注册门槛。

预注册 ability-improvement gate 同时要求：

1. Primary 三种子 mean exact 高于 SFT-only；
2. Challenge 三种子 mean exact 高于 SFT-only；
3. 至少 2/3 seed 同时改善两套；
4. 每套 format 与 mode compliance 都不低于 95%；
5. 训练 family 与两份 holdout 零重叠。

三条路线的 family overlap 都是 `0`，因此只有泄漏门通过；三条
`ability_improvement` gate 全部失败：

- RLVR 与组合路线的 Primary mean 从 `4` 增到 `6`，但 Challenge mean 从
  `6` 降到 `5.33`；
- direct-RLAIF 的 Challenge mean 小幅增到 `6.67`，Primary mean 却降到
  `3`，且出现 `0, 6, 3` 的明显 seed 方差；
- 没有任何路线达到至少 `2/3` seed 同时改善两套；
- 三条路线都没有达到 95% format/mode gate。

因此本课不能声称 RLVR、direct-RLAIF 或二者组合提升了可泛化能力，也不选择
最好 seed 量化或部署。正式 checkpoint、provider cache 和逐题 completion
作为 ignored 研究证据保留；Git 中的机器可读摘要只记录哈希、计数和审计指标。

## 10. 怎样理解负结果

若 RLVR 提高 primary 却降低 challenge，最合理解释不是“RL 学会了数学但运气
不好”，而是它放大了少数训练 family 周围已有的输出模式。若 RLAIF 降低 exact
或格式，则说明稀疏 AI 分数没有提供足够稳定的方向，KL 与 clipping 也只能限制
漂移，不能创造缺失的算法表示。

尤其要避免三种 post-hoc 叙事：

- 只报最好 seed；
- 把训练 reward 上升当作 holdout 能力；
- 看到 challenge 回退后临时改 gate、温度或 token cap。

本课保留全部三种子和失败 checkpoint。负结果仍回答了一个有价值的问题：
对于当前 9.5M byte-level policy，512 completions/seed 的 sparse group reward
是否足以稳定改善可泛化数学物理能力。

## 11. 下一步应该做什么

本课结果支持按证据选择后续方向，而不是立刻加大 RL：

1. 若主要瓶颈是零方差，应先提高 base/SFT policy 的 pass@8，让更多组出现
   correct/incorrect 混合，而不是在全错组上重复 policy epoch；
2. 可增加 verifier-friendly curriculum，但必须保持 family-disjoint；
3. 可研究 process reward 或逐步可验证中间状态，避免只有 final 的稀疏信号；
4. direct-RLAIF 可改为先收集 preference dataset、训练本地 reward model 的
   独立路线，但必须与本课 direct reward 分开命名；
5. DPO 是 preference optimization control，不应和 RLVR 混称；
6. 只有多种子 holdout 通过 gate 后，才值得重新量化、导出 C 并上 Nspire。

## 12. 复现实验

先做双起点 screen：

```powershell
python -m nanogpt_nspire.lesson17_start_screen `
  --v1-checkpoint <lesson15.pt> --v1-sha256 <sha256> `
  --v2-checkpoint <lesson16-v2.pt> --v2-sha256 <sha256> `
  --primary-evaluation <evaluation.jsonl> `
  --challenge-evaluation <challenge_evaluation.jsonl> `
  --output-dir artifacts/lesson17-start-screen `
  --device cuda
```

正式训练的三种 route 分别使用：

```powershell
python -m nanogpt_nspire.group_policy_train `
  --route rlvr `
  --seed 20260731 `
  --start-checkpoint <selected.pt> `
  --start-checkpoint-sha256 <sha256> `
  --start-route GQA-ALiBi-Hybrid-SFT-Context512 `
  --primary-evaluation <evaluation.jsonl> `
  --challenge-evaluation <challenge_evaluation.jsonl> `
  --output-dir artifacts/lesson17-formal/rlvr-seed-20260731 `
  --source-commit <commit> `
  --device cuda
```

`rlaif` 与 `combined` 还必须显式提供 ignored judge cache：

```text
--judge-cache artifacts/lesson17-formal/alias-rlaif-cache-seed-20260731.jsonl
```

最后每个 checkpoint 用 `nanogpt_nspire.lesson17_eval` 评测，并用
`nanogpt_nspire.lesson17_report` 生成固定 claim gate 报告。生成数据、原始
completion、provider cache、checkpoint 与评测 JSON 全在 ignored
`artifacts/`；Git 只保存代码、Lesson 和去掉原始文本的机器可读汇总。
