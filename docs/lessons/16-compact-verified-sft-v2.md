# Lesson 16：紧凑可验证 SFT v2、终止边界与“自洽但算错”

Lesson 15 的 GQA-ALiBi Think 路线在 256-token 上只有 `75.78%` 会正确进入
`<FINAL>`，`20.31%` 会耗尽输出预算；Direct 与 Think 都只答对 `2/128`。
本课先不做 RL，而是把两个问题拆开：

1. 用更短、逐步精确验证的监督数据和边界加权 CE，能否先教会模型稳定结束？
2. 格式稳定以后，冻结数学物理题与新挑战题的数值正确率是否真的提高？

机器可读结果与预先冻结的协议位于：

- [`lesson16-compact-verified-sft-v2.json`](../../experiments/lesson16-compact-verified-sft-v2.json)
- [`Lesson 16 implementation plan`](../plans/2026-07-29-lesson16-sft-v2.md)
- [`Lesson 16 RLVR/RLAIF design`](../plans/2026-07-29-lesson16-sft-v2-rlvr-rlaif-design.md)

## 1. 这仍是 SFT，不是 RL、RLAIF 或蒸馏

本课唯一训练目标仍是 token-level supervised cross-entropy。它没有：

- 从 DeepSeek 或其他外部 API 生成新答案；
- 读取 teacher logits 或概率分布；
- rollout、reward、advantage、policy update 或 preference pair；
- RLVR、RLAIF、DPO、GRPO、PPO 或 reward model。

因此准确名称是：

```text
compact verified SFT v2
```

数据中保留了一份故意写错代入或 final 的 hard-negative JSONL，但普通 SFT
只会从“被 loss mask 选中的正确 assistant target”学习。没有 preference loss
时，把错误答案直接混进正样本不会让模型“知道它是错的”，只会监督模型复现错误。
所以 hard negative 在本课严格标记为 `training_eligible=false`，只用于评测和
Lesson 17 的 verifier/preference 实验。

## 2. 从 GSM8K 注释制备紧凑、可复算步骤

GSM8K 的答案中常含：

```text
<<12*7=84>>
```

本课不执行字符串，也不调用 Python `eval`。专用递归下降解析器只接受：

```text
finite integer/decimal
+ - * /
unary +/-
bounded parentheses
```

随后用 `Fraction` 做精确运算，只接受可表示为有限小数的结果，并逐项检查
`expression == declared result`。每条 GSM8K 还必须满足：

- 有 1–4 个计算注释；
- 最后一个注释结果等于独立解析的 `#### final`；
- reasoning 不超过 160 UTF-8 bytes；
- Direct 与 Think 编码都能装入 512 context；
- 不与主评测或 Lesson 16 挑战 family 重叠。

最终 reasoning 形如：

```text
Compute 12 * 7 = 84.
```

这与“让 teacher 自由写一大段 CoT”不同：每一步都来自数据集的显式计算注释，
并能被本地解析器重新计算。

### 2.1 为什么没有直接复用旧生成器和旧 256-token GSM 门

第一次正式构建暴露了两个数据工程问题：

1. Lesson 10–14 的算术生成空间适合 12,000 family，但在固定路线分配下不足以
   稳定生成 24,000 个唯一 family；
2. 旧 `GSM8KExample` 在压缩 reasoning 以前就用 256-token conversation 门
   拒绝长题，这与本课 512-context 合同不一致。

修复没有改变旧课：Lesson 16 新增更大的均衡算术域，并复用相同规范化与 family
ID 规则，在紧凑 reasoning 形成以后再由 512-context packer 判定长度。最终仍有
62 个 family 因 Direct 超长、187 个因 Think 超长被明确拒绝，而不是静默截断。

## 3. 数据规模、拒绝原因与确定性

输入与接受结果：

| source | input/target | compact verified accepted |
|---|---:|---:|
| project arithmetic | 24,000 | 24,000 |
| project numeric physics | 12,000 | 12,000 |
| pinned GSM8K train | 7,473 | 5,402 |
| total | 43,473 | 41,402 |

GSM8K 的 2,071 个拒绝包括：

| reason | count |
|---|---:|
| calculation count is not 1–4 | 1,333 |
| unsupported calculation token | 385 |
| last calculation differs from final | 315 |
| result is not a decimal literal | 37 |
| expected a number | 1 |

排除主评测、挑战集和超长 family 后，正式 corpus 为：

| split fact | value |
|---|---:|
| unique families | 41,122 |
| Direct + Think records | 82,244 |
| packed tokens | 8,881,031 |
| train / validation / test families | 36,978 / 2,100 / 2,044 |
| max / mean reasoning bytes | 136 / 42.03 |

两次独立构建的全部 10 个文件逐字节一致。关键 hash：

```text
root manifest
0837e63e7003c5250f2c2d7cfb486d9388b597da2b10f1fe2b06ad5be141759e

packed corpus manifest
9f45ca7cc079d7cbf64e1e53c0cf7cb584583a65aacc511410f0b8e02c954365

challenge evaluation
359d24949bbe74ff7ee2a88fb9d70a8dc5f4bd50d569883de2306446538a2b00

hard negatives
f802953412deea0576583e368e64bb66a4a66ff2bbd41368dd5f76727b04296c
```

## 4. 为什么给 `<FINAL>` 和 `<EOS>` 更高权重

普通 masked CE 是所有 eligible assistant token 的平均：

```text
CE = sum(token_ce) / count(tokens)
```

reasoning 有几十个 byte token，而 `<FINAL>` 和 `<EOS>` 各只有一个。即使模型在
边界 token 上反复出错，普通平均也可能被大量容易的正文 token 稀释。本课使用：

```text
weight = 4, target is <FINAL> or <EOS>
weight = 1, other eligible assistant targets

loss = sum(token_ce * weight) / sum(weight)
```

Direct prompt 已由控制格式预置 `<FINAL>`，所以 Direct 训练主要额外强调生成
`<EOS>`；Think 必须自己生成 `<FINAL>` 与 `<EOS>`，两者都被强调。

这不是 reward。目标 token 仍由数据给定，梯度仍来自加权 cross-entropy。为了
保持与旧 checkpoint 公平，选模和完整 validation/test 仍使用未加权 CE。

## 5. 固定训练合同与 loss

架构完全继承 Lesson 15：

| field | value |
|---|---:|
| layers / Q heads / K/V heads | 6 / 6 / 2 |
| width / context / vocabulary | 384 / 512 / 264 |
| parameters | 9,543,552 |
| theoretical FP32 KV cache | 3,145,728 bytes |
| parent route | `GQA-ALiBi-Hybrid-SFT-Context512` |
| new route | `GQA-ALiBi-SFT-v2-Context512` |

训练合同：

| field | value |
|---|---:|
| optimizer updates | 1,000 |
| tokens/update | 4,096 |
| sampled tokens | 4,096,000 |
| seed | 20260729 |
| max/min LR | `1e-4 / 1e-5` |
| warmup / eval interval | 50 / 100 |
| boundary weight | 4 |

结果：

| metric | value |
|---|---:|
| initial full validation CE | 1.134604 |
| selected full validation CE | 0.246123 |
| selected full test CE | 0.245974 |
| best step | 1,000 |
| wall time | 62.38 s |
| update throughput | 66,662 sampled tokens/s |
| peak CUDA allocation | 447,446,016 B = 426.72 MiB |

checkpoint：

```text
38,592,622 bytes
SHA-256 e78c78bfbf8a7df1c8c623d27300084d8b67d14827c69fdc523b8b4df1a299ed
```

很低的 validation CE 只说明模型很会复现本课分布中的短模板与常见 token。
它不能替代未见题的 exact verifier。

## 6. 主评测：终止问题解决，算题没有进步

主评测仍是 Lesson 12 冻结的 128 个 family，每类 32 题，greedy decode，
Direct/Think 都给 256 个输出 token。

| route | mode | exact | format/mode | `<FINAL>` | budget trunc. | mean reasoning |
|---|---|---:|---:|---:|---:|---:|
| parent | Direct | 2/128 | 100.00% | — | 0.00% | 0.00 |
| SFT v2 | Direct | 2/128 | 100.00% | — | 0.00% | 0.00 |
| parent | Think | 2/128 | 75.78% | 75.78% | 20.31% | 75.23 |
| SFT v2 | Think | 2/128 | 100.00% | 100.00% | 0.00% | 38.26 |

因此预注册的主门全部通过：

- Direct 格式/模式保持 100%；
- Think 格式、模式和 `<FINAL>` 超过 75.78%，并达到 100%；
- Think budget truncation 从 20.31% 降到 0%；
- Direct/Think exact 都没有低于父模型的 2/128；
- 95% Think completion 的 aspirational target 也达到。

但 exact 完全没有提高。SFT v2 成功解决的是“怎么结束”，不是“怎么算对”。

## 7. 新挑战集：主评测持平掩盖了范围迁移退化

挑战集有 256 个从未训练的 family，每个切片 64 题：

- `in_range`：与训练分布同范围的新 family；
- `range_shifted`：更大数量级；
- `sign_shifted`：更多负数与符号组合；
- `substitution_adversarial`：物理量次序和数值组合容易诱发错误代入。

同一 evaluator、greedy decode、256-token cap：

| route | mode | exact | in-range | range-shift | sign-shift | substitution |
|---|---|---:|---:|---:|---:|---:|
| parent | Direct | 3/256 | 1/64 | 0/64 | 2/64 | 0/64 |
| parent | Think | 3/256 | 1/64 | 0/64 | 2/64 | 0/64 |
| SFT v2 | Direct | 1/256 | 0/64 | 0/64 | 1/64 | 0/64 |
| SFT v2 | Think | 1/256 | 0/64 | 0/64 | 1/64 | 0/64 |

四次挑战评测的 format、mode、unit 与 EOS 均为 100%，budget/context truncation
均为 0%。因此退化不能归因于“答案没写完”或“漏单位”，而是数值本身不对。

按生成 token 归一化也没有改变结论：

| route | mode | correct / 1000 generated tokens |
|---|---|---:|
| parent | Direct | 0.5211 |
| parent | Think | 0.2098 |
| SFT v2 | Direct | 0.1699 |
| SFT v2 | Think | 0.0607 |

## 8. 最关键现象：自洽、复制题目数字，仍然可以算错

Think challenge 的额外诊断：

| diagnostic | parent | SFT v2 |
|---|---:|---:|
| output uses at least one prompt number | 19.53% | 98.83% |
| last reasoning number equals final number | 26.56% | 64.06% |
| exact answer | 1.17% | 0.39% |
| unique finals | 73 | 193 |
| most common final concentration | 13.28% | 3.13% |

SFT v2 明显减少了单一 final 吸引盆，也更常读取题目数字，reasoning/final 更常
局部自洽；但这些改进没有变成正确计算。典型失败可以满足：

```text
uses a number from the prompt
looks like formula substitution
reasoning and final repeat the same value
terminates cleanly
```

同时仍然把错误的量代入公式，或把两个正确操作模板拼成错误计算。这说明：

```text
format compliance != mathematical correctness
prompt-number copying != correct variable binding
reasoning/final consistency != verifier correctness
low SFT loss != out-of-family generalization
```

它与早期 `the state` 文本吸引盆属于同一类警告：模型可以学到强局部模式，
而没有学会我们希望的条件算法。

## 9. Lesson 17 应该怎样接

本课的预注册主格式门通过，但挑战 exact 从 `3/256` 降到 `1/256`。不能在看到
结果以后悄悄修改原门槛，也不能只报主评测持平就称 v2 更强。

因此保存两个起点：

```text
parent: better challenge exact, weaker Think termination
SFT v2: perfect termination, worse challenge exact
```

Lesson 17 应先做双起点的 verifier-guided 小规模实验，再决定 RL 主线：

1. 用本地 exact value、unit、EOS 和格式 verifier 做 RLVR；
2. 把 prompt-number binding、reasoning/final consistency 作为诊断或辅助 reward，
   但绝不能替代最终 exact reward；
3. DeepSeek AI feedback 只评价解释清晰度和概念合理性，独立于数值 verifier；
4. SFT-only、RLVR、RLAIF、RLVR+RLAIF 使用相同 prompt family、rollout token
   和多 seed；
5. 同时报告 Direct/Think 相同 token cap 下的准确率和 accuracy/token；
6. 若要研究 CoT 是否有益，必须比较相同输出 token 限制，不能让 Think 偷用
   更多预算。

## 10. 复现

```powershell
$env:PYTHONPATH = (Resolve-Path training).Path

python -m nanogpt_nspire.lesson16_data `
  --gsm8k-train artifacts/lesson12-downloads/gsm8k-train.jsonl `
  --evaluation artifacts/lesson12-data-v2/evaluation.jsonl `
  --output-dir artifacts/lesson16-data-v2 `
  --registry-path experiments/lesson10-public-sources.json `
  --arithmetic-count 24000 `
  --physics-count 12000 `
  --seed 20260729

python -m nanogpt_nspire.sft_v2_train `
  --data-dir artifacts/lesson16-data-v2/hybrid_512 `
  --output-dir artifacts/lesson16-sft-v2 `
  --parent-checkpoint artifacts/gqa-alibi-sft/gqa_alibi_hybrid_sft_context512.pt `
  --parent-checkpoint-sha256 f385a51f7ac4d53b1a640f9a977308a365bb032a6a2dbe53394ba035169513ee `
  --source-commit 10af501 `
  --device cuda
```

`reasoning_eval` 负责 128 题主门，`lesson16_eval` 负责全部 256 个挑战 family
及逐切片诊断。

## 11. 声明边界

- 本课完成的是 PyTorch SFT v2 数据、训练和 Host GPU 评测。
- 没有完成 teacher-logit 蒸馏、RL、RLVR、RLAIF 或 preference optimization。
- 38.6 MB 是 FP32 PyTorch checkpoint，不是 Nspire 部署文件。
- 没有导出量化 GQA、实现 GQA/ALiBi C kernel、做 PyTorch/C logits 对齐、
  Ndless build 或 CX II 真机内存/速度测试。
- 本课不能证明当前 9.5M byte-level model 是可靠数学物理助手；挑战集结果明确
  证明它仍不可靠。
