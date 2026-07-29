# Lesson 15：256-token 推理、GQA、ALiBi 与计算器边界

Lesson 14 已经把 context 从 256 延长到 512，并证明 96-token 诊断能让更多
CoT 写完，却没有提高冻结题准确率。本课继续回答两个问题：

1. Direct 与 Think 都放宽到至少 256 个输出 token 后，CoT 会不会终于提高
   数学物理准确率？
2. 512 context 的 MHA FP32 KV cache 要 9 MiB，能不能用 2-group GQA 降到
   3 MiB，并比较 learned position 与 ALiBi？

机器可读结果、设计与决策记录：

- [`lesson15-long-output-gqa-alibi.json`](../../experiments/lesson15-long-output-gqa-alibi.json)
- [`Lesson 15 design`](../plans/2026-07-29-lesson15-long-output-gqa-alibi-design.md)
- [`Lesson 15 implementation plan`](../plans/2026-07-29-lesson15-long-output-gqa-alibi.md)
- [`ADR-0001`](../adr/0001-gqa-alibi-long-context-prototype.md)

## 1. 为什么 256 是主预算，384 只是诊断

冻结评测仍是 Lesson 12 的 128 题，每类 32 题，greedy decode。最长输入为
233 token，所以 512 context 至少留下：

```text
512 - 233 = 279 output positions
```

因此 Direct 与 Think 都用 256-token cap 时，每道题都有相同可用输出容量，且
不会因为 context 先耗尽而受到不公平截断。384 超过某些题的剩余 context，只能
作为“继续加预算会发生什么”的诊断，不能作为主比较。

这里的 token 仍是 byte tokenizer token。256 token 大致只能容纳 256 个英文
UTF-8 byte，不等于成熟子词模型的 256 token。

## 2. 256-token MHA 基线：写得更长仍不等于想得更对

Lesson 14 的 `Hybrid-Control-SFT-Context512` 在新预算下：

| mode | exact | format | `<FINAL>` transition | budget trunc. | context trunc. | mean reasoning |
|---|---:|---:|---:|---:|---:|---:|
| Direct-256 | 2/128 | 100.0% | — | 0.0% | 0.0% | 0.0 |
| Think-256 | 1/128 | 75.0% | 75.0% | 17.97% | 0.0% | 68.80 |

相较 96 token，256 消除了 context truncation，并允许模型平均写出约 69 个
reasoning token；但精确率从 2/128 变为 1/128，而不是提高。

384-token Think 进一步把 budget truncation 从 17.97% 降到 0.78%，同时让
context truncation 升到 17.19%，准确率仍是 1/128。它说明 256 不是任意选择：

```text
96   -> completion space often too short
256  -> no context truncation on the frozen set
384  -> trades budget truncation for context truncation
```

更重要的是，它再次排除了“只要让 CoT 多写一点就会正确”这个解释。当前模型会
生成更完整的公式模板和数值代入句，但经常代入错误数字、做错运算，或者 final
与 reasoning 自相矛盾。

## 3. GQA 到底压缩了什么

MHA 为每个 query head 各保存一组 K/V。当前模型：

```text
6 query heads
6 K heads
6 V heads
head_dim = 64
```

本课的 2-group GQA 改成：

```text
6 query heads
2 K heads
2 V heads
3 query heads share one K/V group
```

512 context、6 layers、FP32 cache 的理论字节为：

```text
2 * layers * context * kv_heads * head_dim * sizeof(float)
```

| architecture | K/V heads | FP32 KV at 512 |
|---|---:|---:|
| MHA | 6 | 9,437,184 B = 9.00 MiB |
| GQA | 2 | 3,145,728 B = 3.00 MiB |

所以 GQA 真正直接减少的是 K/V projection 参数、未来增量 decode 的 KV cache
和相关内存带宽；它没有减少 6 个 query head 的 attention score 计算。

[GQA 论文](https://arxiv.org/abs/2305.13245)研究了把已有 MHA checkpoint
uptrain 成少量 K/V head，并在其模型尺度上取得接近 MHA 的质量与接近 MQA 的
速度。那是实验动机，不是对 10M byte model、250-step uptraining 的保证。

## 4. 从 MHA checkpoint 转成 GQA

两条 GQA 路线都从完全相同的 `Math-Physics-CPT-Context512` 开始：

```text
SHA-256
556bb6d7377f8eb69b5a77ac4c6ae084482bb76359732cc5f322eb3e21545548
```

转换过程为：

1. Q projection 逐元素复制；
2. 把 6 个 K head 按连续 3 个一组求均值，得到 2 个 K head；
3. V 做相同分组平均；
4. token embedding、normalization、attention output、MLP 与 tied LM head
   全部逐元素复制；
5. GQA-Learned 保留 512-row position table；
6. GQA-ALiBi 删除 learned position table，使用固定 per-query-head slope。

这不是无损转换。三组 K/V 的均值会丢掉 head-specific 信息，所以转换后完整
validation loss 会明显跳高：

| route | just converted full val loss | after 250-step CPT |
|---|---:|---:|
| GQA-Learned | 2.6654 | 1.6191 |
| GQA-ALiBi | 2.5183 | 1.5890 |

250-step、1,024,000-token CPT 恢复了大部分语言建模质量，但“恢复大部分”不等于
与父 MHA 完全相同。

### 4.1 为什么同时记录文件 hash 与 tensor-state hash

最初的确定性测试发现：相同权重两次 `torch.save` 的 `.pt` 文件 SHA-256
不一致。原因是 PyTorch zip container 的内部 storage 编排不属于规范化序列化
合同；这不代表 tensor 数值不同。

因此本课加入 canonical `model_state_sha256`：

```text
sorted tensor name
  + dtype
  + shape
  + contiguous value bytes
```

两次独立转换的 canonical state hash 完全一致。`.pt` 文件 hash 仍用于验证某个
具体 artifact 没被替换，state hash 则用于验证“模型数值状态相同”。这两个
问题不能混为一个。

## 5. Learned position 与 ALiBi

GQA-Learned 延续绝对位置表：

```text
hidden = token_embedding + position_embedding[position]
```

GQA-ALiBi 不再加入位置向量，而在每个 query head 的 attention score 上加入
与距离成正比的负 bias。6 个 head 的固定 slope 为：

```text
0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125
```

[ALiBi 论文](https://arxiv.org/abs/2108.12409)研究了 train-short/test-long
行为。本课只在已经训练过的 512 window 内比较前后半区，不声称完成 512 之外
的长度外推：

| CPT route | positions 0–255 | positions 256–511 | all |
|---|---:|---:|---:|
| GQA-Learned | 1.6075 | 1.6335 | 1.6205 |
| GQA-ALiBi | 1.6048 | 1.5761 | 1.5904 |

Learned 的后半区比前半区略差；ALiBi 的后半区反而略好，并取得较低的 overall
loss。这是支持继续研究 ALiBi 的证据，但还不是外推证据。真正测试
train-512/test-long，需要模型、评测器和部署 runtime 都允许超过 512。

## 6. 同预算 Hybrid SFT

两条 GQA 路线均冻结：

| contract | value |
|---|---:|
| optimizer updates | 1,000 |
| tokens/update | 4,096 |
| sampled tokens | 4,096,000 |
| seed | 20260728 |
| context | 512 |
| Direct/Think family and format | identical |

结果：

| route | parameters | full val | full test | train tok/s |
|---|---:|---:|---:|---:|
| MHA-Learned | 10,919,808 | 0.8282 | 0.8531 | 73,849 |
| GQA-Learned | 9,740,160 | 0.8554 | 0.8769 | 57,444 |
| GQA-ALiBi | 9,543,552 | 0.8196 | 0.8452 | 46,630 |

在本轮数据上：

- GQA-Learned 比 MHA loss 略差；
- GQA-ALiBi 的 validation/test loss 略优于 MHA；
- 两条 GQA 的 PyTorch 训练吞吐都更低，ALiBi 尤其如此。

这不矛盾。当前教学实现为了保持明确、可移植的张量语义，会在 attention 计算前
把 2 组 K/V broadcast 到 6 个 query head；权重与未来 cache 更小，但 GPU
full-sequence kernel 不一定更快。

## 7. 256-token 三架构冻结评测

| route | mode | exact | format / mode | budget trunc. | context trunc. | mean reasoning |
|---|---|---:|---:|---:|---:|---:|
| MHA-Learned | Direct | 2/128 | 100.0% | 0.00% | 0.00% | 0.00 |
| MHA-Learned | Think | 1/128 | 75.00% | 17.97% | 0.00% | 68.80 |
| GQA-Learned | Direct | 2/128 | 100.0% | 0.00% | 0.00% | 0.00 |
| GQA-Learned | Think | 2/128 | 75.78% | 15.63% | 0.00% | 63.80 |
| GQA-ALiBi | Direct | 2/128 | 100.0% | 0.00% | 0.00% | 0.00 |
| GQA-ALiBi | Think | 2/128 | 75.78% | 20.31% | 0.00% | 75.23 |

不能把 `1/128 -> 2/128` 写成 GQA 让 CoT 翻倍。只有一题差异，样本太小，而且
三条 checkpoint 的训练轨迹不同。稳健结论是：

1. 三种架构的 Direct 都只有 2/128；
2. Think 都没有超过 2/128；
3. GQA 把理论 KV cache 降到三分之一，没有观察到灾难性 exact regression；
4. 更低的 SFT loss 没转化为可用的数值推理准确率；
5. 256-token Think 仍有约 16%–20% 不生成 EOS 而耗尽预算。

输出也出现明显数字吸引盆。例如很多不同题会落到 `10`、`11.5`、`12`、`13`
等训练中常见数字。模型能复制“Use F = m a. Substitute ...”这类形状，却常把
prompt 里的数字换成记忆中的高频数字。这与早期 Tiny Shakespeare 的
`the state` 吸引盆是同一类现象：低 loss 的局部文本模式不等于输入条件被正确
使用。

## 8. 为什么没有保留 PyTorch 原生 GQA kernel

PyTorch 2.11 提供
`scaled_dot_product_attention(..., enable_gqa=True)`。本课做了一个未提交的
主机诊断：batch 2、sequence 512、10 次 warmup、50 次 full forward。

| position | explicit reference | native GQA | change |
|---|---:|---:|---:|
| learned | 220,286 tok/s | 207,818 tok/s | -5.66% |
| ALiBi | 198,732 tok/s | 173,942 tok/s | -12.47% |

这项主机 kernel 替换数值测试通过，却在该形状与硬件上更慢，所以未提交并撤回。
更根本的原因是：TI-Nspire 不会运行 PyTorch 或 CUDA kernel。计算器需要我们
自己的 scalar/NEON C 实现。可迁移的部分是：

- Q/K/V 的 GQA tensor layout；
- 2-group K/V 权重；
- 3 MiB cache 预算；
- ALiBi score 公式；
- 量化后的 C 算子与实际 Ndless 测量。

主机训练 kernel 是否快，不能代替 Nspire 上的 C decode 证据。

## 9. 现在要继续加强 SFT，还是直接 RLAIF

答案是：**先做 SFT v2，再以 RLVR 为数值主线，把 DeepSeek-RLAIF 用在难以
程序验证的质量维度。**

当前 Think mode 的格式/模式合规只有约 76%，并且约五分之一会耗尽 256-token
预算。直接 RL 会把两个问题混在一起：

```text
模型不会稳定终止
        +
模型不会可靠计算
```

SFT v2 应先改善：

1. 更一致的 `<THINK> reasoning <FINAL> answer <EOS>` 边界；
2. 对当前 byte model 足够短的 1–4 step rationale；
3. 明确监督“读取 prompt 数字 -> 公式 -> 代入 -> 运算 -> final”；
4. 更均衡的数值、符号、单位、数量级和负数分布；
5. hard-negative 示例：公式正确但代入错误、reasoning 正确但 final 错误；
6. Direct 与 Think 分层采样，避免模型只学到一个模式；
7. 终止与 repetition 约束，防止用更长 CoT 掩盖不确定性。

[DeepSeek-R1](https://arxiv.org/abs/2501.12948)提供了相同的警告：
R1-Zero 的纯 RL 路线出现重复、可读性和语言混合问题；R1 使用 cold-start
data 与多阶段训练。本项目规模远小于 R1，更不能假设纯 RL 会自动整理输出。

### 9.1 数学与数值物理优先 RLVR

对可计算题，reward 可由本地确定性 verifier 给出：

```text
exact numeric value
unit equivalence
format / EOS
no role-token leakage
bounded length
optional formula identity
```

这类 reward 不需要 DeepSeek 猜答案，也不受 judge 文风偏好影响。
[DeepSeekMath](https://arxiv.org/abs/2402.03300)提出 GRPO 并用于数学推理；
但我们的 10M 模型是否适合 GRPO 必须实测，不能照搬 7B 结果。

### 9.2 DeepSeek-RLAIF 负责什么

RLAIF 的严格含义不是“让 DeepSeek 生成训练答案”。在
[Constitutional AI](https://arxiv.org/abs/2212.08073)中，AI 对候选输出给出
偏好，训练 preference/reward model，再用该 reward 做 RL。

本项目可以让 DeepSeek 按冻结 rubric 比较两条候选解释：

- 哪条更清楚；
- 物理概念是否合理；
- 是否使用题中真实数值；
- reasoning 与 final 是否一致；
- 是否不必要地冗长；
- 非数值概念题是否有关键解释。

但最终数值正确性仍由程序 verifier 决定。否则 judge 可能偏爱一条流畅但算错的
解释，student 也可能学会 reward hacking。

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)还说明，在其 MATH
研究设置中，step-level process supervision 优于只看最终结果。这支持我们把
公式、代入和计算分解验证；它同样不能直接保证 10M student 的收益。

### 9.3 后续公平对照

下一阶段冻结四条相同起点、相同 rollout token、相同 prompt family 的路线：

| route | deterministic verifier | DeepSeek preference |
|---|---:|---:|
| SFT-only | no | no |
| SFT + RLVR | yes | no |
| SFT + RLAIF | final score only | yes |
| SFT + RLVR + RLAIF | yes | yes |

所有路线同时报告：

- Direct/Think 在相同 256-token cap 下的 exact；
- format、EOS、budget/context truncation；
- mean reasoning tokens 与 accuracy-per-generated-token；
- 未见 family、数值范围外、单位变体与 adversarial substitution；
- reward 与独立 holdout accuracy 是否脱钩；
- 多 seed 均值，而不是只挑最好的一次；
- API 反馈属于 AI preference，不写成 logit distillation。

## 10. Nspire 部署边界

本课还没有“在计算器上支持 GQA/ALiBi”。当前产物只在 PyTorch/CUDA 上通过：

- tensor shape 与参数公式测试；
- MHA→GQA 转换不变量；
- canonical tensor-state hash；
- CPT/SFT 训练；
- position split loss；
- 256-token frozen generation。

要称为 Nspire 架构，还必须完成：

1. 新 NGM schema，声明 `n_kv_head` 与 position mode；
2. packed INT4/W4A8 GQA exporter；
3. Host C GQA + ALiBi logits/greedy sequence 对齐；
4. 增量 KV cache，不在启动时展开成 MHA FP32；
5. Ndless build/package；
6. 真机模型打开、256-token 上限、峰值 RAM、TTFT 和 token/s；
7. Exit 后 cache 清零与界面恢复。

因此 Lesson 15 的准确结论是：

> 在 512-context、10M byte-level student 上，256-token 输出预算消除了冻结集的
> context truncation，却没有让 CoT 获得可靠数学推理；2-group GQA 将理论 FP32
> KV cache 从 9 MiB 降到 3 MiB，ALiBi 在本轮 512-window loss 上表现最好，
> 但两者仍需新的量化 C runtime 与真机验证。下一步应先做 SFT v2，再把可验证
> 数值奖励与 DeepSeek AI preference 分开实验。
