# Lesson 14：可控短 CoT、固定 token 公平比较与 512 context

Lesson 13 证明了两件不同的事：外部 API 文本属于合成数据 SFT，本地完整
logits 才属于严格蒸馏；但无论 hard target 还是 soft target，10.8M student
都没有凭空学会可泛化算术。本课再问两个更窄的问题：

1. 明确写出短推理过程，能否在相同输出 token 上限下提高精确答案？
2. 同一个模型能否通过一个精确 control token 切换“直接答”和“先想再答”？

同时增加一个独立扩展实验：把 context 从 256 延长到 512，测清楚位置外推、
继续预训练和 Nspire KV 内存分别发生什么。CoT 与 context 不混成一个主实验，
否则结果变化无法归因。

机器可读合同与结果：

- [`lesson14-controllable-cot-and-context.json`](../../experiments/lesson14-controllable-cot-and-context.json)
- [`Lesson 14 design`](../plans/2026-07-29-lesson14-controllable-cot-design.md)
- [`Lesson 14 implementation plan`](../plans/2026-07-29-lesson14-controllable-cot.md)

## 1. 这次仍是 SFT，不是 RL，也不是蒸馏

本课每个 assistant target 都在训练前固定，优化目标仍是 assistant-only
cross entropy：

```text
fixed demonstration -> next-token CE -> parameter update
```

没有在线采样、reward、verifier reward、policy ratio、KL-to-reference policy 或
preference pair，所以不是 RLVR、PPO、GRPO 或偏好优化。也没有让 student 学习
teacher 的完整条件概率分布，所以不是 logit distillation。

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 正好说明了边界：R1-Zero
直接用 RL 获得推理行为，但出现可读性和语言混合问题；R1 再加入 cold-start
数据和多阶段训练。我们在本课只研究其中的 **cold-start / reasoning-format
SFT** 部分，RLVR 留到后续课程。论文中的最小公开 distilled dense model 仍是
十亿参数级，不能把它的结果直接外推到我们的 10.8M byte model。

## 2. 用 system prompt 控制，还是用特殊 token 控制？

当前 264-token 词表已经预留：

```text
261  <THINK>
262  <FINAL>
```

因此不需要改 tokenizer。两种序列为：

```text
Direct
<BOS><USER>question<ASSISTANT><FINAL>answer<EOS>

Short CoT
<BOS><USER>question<ASSISTANT><THINK>reasoning<FINAL>answer<EOS>
```

推理时 `<FINAL>` 或 `<THINK>` 由程序作为 assistant prefix 放入。Direct 模型
只需生成答案和 `<EOS>`；CoT 模型必须自己生成 reasoning、`<FINAL>`、答案和
`<EOS>`。

这比只写自然语言 system prompt 更适合作为实验变量。自然语言 prompt 也是普通
token 序列；除非训练语料反复展示完全一致的控制语义，否则“please think”不一定
可靠。专用 token 则只有一个位置、没有措辞歧义，而且评测可以精确判断模型是否
完成 `<THINK> -> <FINAL>` 状态转换。

这并非凭空发明的界面技巧。官方
[DeepSeek-V3.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
通过不同 chat-template prefix 让同一模型进入 thinking 或 non-thinking 模式；
当前 [DeepSeek API thinking-mode 文档](https://api-docs.deepseek.com/guides/thinking_mode/)
也把 thinking 作为显式请求开关。我们的 `<THINK>` / `<FINAL>` 是同一思想在固定
264-token 教学模型上的最小实现，不复制其模型规模或私有训练配方。

需要特别区分：

- **模式控制成功**：模型能按 cue 输出指定结构；
- **推理能力成功**：结构中的中间步骤真实、正确，并提高未见题精度。

前者可以由 SFT 很快学会；后者不一定随格式一起出现。

## 3. loss mask 到底监督哪些 token

Direct target：

```text
<BOS><USER>question<ASSISTANT><FINAL>answer<EOS>
  0      0        0          0       1...1   1
```

`<FINAL>` 是推理时已经给出的 cue，因此不计 loss；只训练 answer 与 `<EOS>`。

CoT target：

```text
<BOS><USER>question<ASSISTANT><THINK>reasoning<FINAL>answer<EOS>
  0      0        0          0       1...1     1     1...1   1
```

`<THINK>` 是已给 cue；reasoning、模型要生成的 `<FINAL>`、answer 和 `<EOS>`
全部计 loss。评测时只解析生成的 `<FINAL>` 之后的 final segment。即使 reasoning
里碰巧出现正确数字，只要 final segment 错误或不存在，也记 0 分。

对应实现：

- [`reasoning_format.py`](../../training/nanogpt_nspire/reasoning_format.py)
- [`reasoning_eval.py`](../../training/nanogpt_nspire/reasoning_eval.py)

## 4. 数据如何做到可核验和无泄漏

数据源只有三类：

- 项目生成的精确 arithmetic；
- 项目生成的入门 physics，包含公式、数值代入、答案与单位；
- GSM8K train 的公开 worked solution；删除 `<<...>>` annotation，最终数值再用
  独立 parser 核验。

所有 Direct/CoT 变体共用同一个 `family_id`，所以同题不会跨
train/validation/test。Lesson 12 冻结评测集的 785 个 family 全部排除，任何
命中都会让构建失败。

正式构建接收 20,337 个 family。由于 CoT 序列更长：

| Corpus | context | family | record | token |
|---|---:|---:|---:|---:|
| Direct-256 | 256 | 15,999 | 15,999 | 961,360 |
| CoT-256 | 256 | 15,999 | 15,999 | 1,428,538 |
| Hybrid-256 | 256 | 15,999 | 31,998 | 2,389,898 |
| Hybrid-512 | 512 | 19,283 | 38,566 | 4,253,758 |

两次独立构建的 root manifest 与抽查的 token/mask 文件 SHA-256 全部一致。256
三路线使用完全相同的 15,999 个 family；Hybrid 为每题各放一个 Direct 与 CoT
record。

这批 GSM8K rationale 是公开人工数据，不是本课调用 DeepSeek 生成的文本。本课
也没有把 Lesson 13 的 API `reasoning_content` 当作私有 CoT。即使未来用 API
生成可见推理文本，它仍首先是 hard-label synthetic-data SFT；只有逐 token 学习
共享词表上的 teacher likelihood/logits 才是严格蒸馏。

## 5. 三路线如何公平比较

三条主路线都从同一个 `Math-Physics-CPT` 开始：

```text
ab17a536a58f664f49ff75d176baff7e219996d7d57ce2b6d097eec0b4f89dfb
```

共同冻结：

| 项目 | 固定值 |
|---|---:|
| layers / heads / width | `6 / 6 / 384` |
| parameters | `10,821,504` |
| vocabulary / context | `264 / 256` |
| optimizer updates | `1,000` |
| tokens / update | `4,096` |
| sampled tokens / route | `4,096,000` |
| max/min LR | `1e-4 / 1e-5` |
| seed | `20260728` |
| primary generation | 128 prompts，greedy，最多 48 token |

这叫 **同训练 token / 同计算合同**，不是同 record 数。CoT record 较长，因此
相同 4.096M token 会看到较少的完整 demonstration；Hybrid 还要把预算分给两个
mode。这个取舍本身就是部署预算下 CoT 的真实成本。

训练入口 [`lesson14_train.py`](../../training/nanogpt_nspire/lesson14_train.py)
拒绝修改冻结架构。三条路线都在 step 1000 选中：

| Route | validation loss | test loss | wall time |
|---|---:|---:|---:|
| Direct-Control-SFT | 0.3427 | 0.3494 | 51.4 s |
| Short-CoT-SFT | 0.2839 | 0.2987 | 51.9 s |
| Hybrid-Control-SFT | 0.3092 | 0.3212 | 53.8 s |

这里不能说“CoT loss 最低，所以能力最强”。Direct 的 target 只有短答案；CoT 的
target 包含大量容易预测的模板文字，eligible-token 分布不同。跨目标格式比较
loss 没有清晰的能力含义，必须回到完全相同的冻结题集。

## 6. 48-token 主结果：CoT 没有提高精确率

每个 task 固定抽 32 题，共 128 题：

| Checkpoint | cue | exact | format valid | mode compliant | `<FINAL>` transition | budget truncation |
|---|---|---:|---:|---:|---:|---:|
| Direct | `<FINAL>` | 1/128 | 100.0% | 100.0% | — | 0.0% |
| Short-CoT | `<THINK>` | 0/128 | 45.31% | 64.06% | 67.19% | 43.75% |
| Hybrid | `<FINAL>` | 2/128 | 100.0% | 100.0% | — | 0.0% |
| Hybrid | `<THINK>` | 2/128 | 44.53% | 62.50% | 64.84% | 40.63% |

最干净的因果比较是同一个 Hybrid checkpoint：

```text
<FINAL>  2/128
<THINK>  2/128
```

参数、训练历史和题目完全相同，只改一个 inference cue。可见 CoT 没有提高本轮
精确答案；它主要消耗了输出预算。Short-CoT 平均先生成 29.39 个 reasoning
token，只剩约 9.35 个 final token。

但 Hybrid 的 cue 并非完全无效：两种 cue 产生明显不同的结构，而且 `<THINK>`
模式约 65% 会尝试进入 `<FINAL>`。因此它学到的是 **controllable reasoning
format**，还不是可靠 reasoning algorithm。

## 7. 96-token 诊断：不是只差一点输出空间

96-token 不是新的主结果，只用来区分 truncation 与能力：

| Checkpoint | cue | exact | format valid | transition | budget trunc. | context trunc. |
|---|---|---:|---:|---:|---:|---:|
| Direct | `<FINAL>` | 1/128 | 100.0% | — | 0.00% | 0.00% |
| Short-CoT | `<THINK>` | 0/128 | 78.13% | 79.69% | 3.13% | 7.81% |
| Hybrid | `<FINAL>` | 2/128 | 100.0% | — | 0.00% | 0.00% |
| Hybrid | `<THINK>` | 2/128 | 77.34% | 77.34% | 3.91% | 4.69% |

更大预算显著改善 CoT 的完整性，却没有增加一题精确答案。这排除了“48 token
太短是唯一失败原因”。剩余解释更可能是：

- 10.8M byte model 容量和基础能力不足；
- SFT 学会了常见解释模板，却没有形成可组合运算算法；
- rationale 可能语言流畅但数值步骤不忠实；
- 字节 token 让 30 个英文字符就消耗 30 token，推理成本尤其昂贵；
- family 覆盖与训练 token 仍不足以支持题型外泛化。

这也是为什么后续 RLVR 必须用 exact verifier 奖励最终答案，并把“训练是否产生
更长 CoT”与“同 token cap 精度是否提高”分开记录。RL 可以优化行为，但不会自动
绕过模型容量和 tokenizer 效率。

## 8. context 不是把 `block_size` 改成 512

当前模型使用 learned absolute position embedding：

```python
token_embedding(token) + position_embedding(position)
```

原 checkpoint 只有 256 行位置表。512 pilot 采用最保守、可审计的初始化：

1. 非位置 tensor 全部逐元素复制；
2. 位置 0–255 逐元素保留；
3. 位置 256–511 复制原 0–255 表作为独立可训练行；
4. 在 512-token CPT window 上继续训练 250 步；
5. 再进行独立 Hybrid-512 SFT。

复制初始化会产生位置别名，但保持原窗口前向行为；继续训练再打破别名。它只新增：

```text
(512 - 256) * 384 = 98,304 parameters
```

位置分段 validation loss 最能说明问题：

| Checkpoint | positions 0–255 | positions 256–511 | all |
|---|---:|---:|---:|
| just initialized | 1.5312 | 2.8666 | 2.1989 |
| after 250-step 512 CPT | 1.5412 | 1.5645 | 1.5528 |

新位置初始 loss 几乎是旧位置的两倍；250 步后两半接近。代价是旧半区 loss
小幅增加约 0.0100。这个结果说明：

```text
larger tensor shape != learned long-context ability
```

512 Hybrid 的短题 Direct 仍为 2/128；Think 在 48/96 token 下为 0/128 和
1/128。它没有显示更长 context 会提升短题推理，而且训练 family 更多、父
checkpoint 也不同，所以不计入三条 256 主路线公平比较。

对应实现：

- [`context_extension.py`](../../training/nanogpt_nspire/context_extension.py)
- [`context_position_eval.py`](../../training/nanogpt_nspire/context_position_eval.py)

## 9. 延长 context 到底要不要换注意力？

答案是：**从 256 到 512，不一定先换；为了 Nspire 上继续扩展，很可能要换其中
一部分。** 要先区分三个瓶颈。

### 9.1 位置表示：模型知不知道“第 400 个 token”在哪里

| 方法 | 适合解决 | 对本项目的代价 |
|---|---|---|
| 扩 learned table + CPT | 最小 512 baseline，本课已做 | 需要长序列训练；长度外推弱 |
| ALiBi | 用 attention bias 做长度外推，无位置表 | 要改每个 head 的 score 和重新训练/适配 |
| RoPE + PI/YaRN | 更成熟的 RoPE 长度扩展 | C 端要做旋转与缩放；当前权重不能无损直转 |

[ALiBi](https://arxiv.org/abs/2108.12409) 展示了 train-short/test-long 的线性
attention bias；[Position Interpolation](https://arxiv.org/abs/2306.15595)
通过缩放 RoPE 位置索引避免直接外推的巨大 attention score；
[YaRN](https://arxiv.org/abs/2309.00071) 进一步改进 RoPE context extension。
这些首先解决 **position validity**，不自动减少 KV cache。

### 9.2 KV cache：计算器能不能存下历史

当前 C runtime 每层为所有 head 保存 FP32 K 和 V：

```text
2 * layers * context * width * 4 bytes
```

| context | MHA FP32 KV |
|---:|---:|
| 256 | 4,718,592 bytes = 4.50 MiB |
| 512 | 9,437,184 bytes = 9.00 MiB |

[MQA](https://arxiv.org/abs/1911.02150) 让所有 query head 共享一组 K/V；
[GQA](https://arxiv.org/abs/2305.13245) 使用少量 K/V group，在质量和内存间
折中；[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 的 MLA 则压缩 KV
latent。对当前 6-head、head_dim 64 模型，理想 MQA 可把 512 KV 从 9.00 MiB
降到约 1.50 MiB；2-group GQA 约 3.00 MiB。它们解决 **KV memory / decode
bandwidth**，不自动延长位置编码。

对 Nspire 而言，下一架构实验最值得优先做的是：

```text
ALiBi or RoPE
        +
2-group GQA (quality-first) / MQA (memory-first)
```

但这会改变 QKV tensor shape 和 C kernel，必须从训练、export、Host C logits
对齐重新走一遍，不能把现有 MHA checkpoint 直接改个 header。

### 9.3 attention 范围：是否必须保留所有旧 token

[Longformer](https://arxiv.org/abs/2004.05150) 使用局部加少量全局 attention；
[Transformer-XL](https://arxiv.org/abs/1901.02860) 用 segment recurrence；
[StreamingLLM](https://arxiv.org/abs/2309.17453) 保留 attention sink 和最近
window 来稳定流式模型。这类方法能让内存随固定 window 而不是总对话长度增长，
但“无限流式”不等于能精确访问全部旧内容：滑出窗口的信息可能丢失。

聊天 UI 可以采用分层策略：

```text
short exact model context
  + recent-turn sliding window
  + optional compact conversation summary
```

本项目退出后不保存对话，所以 summary 也只需在 RAM 中存在并在 Exit 时清零。

### 9.4 FlashAttention 为什么不是计算器答案

[FlashAttention](https://arxiv.org/abs/2205.14135) 和
[FlashAttention-2](https://arxiv.org/abs/2307.08691) 主要减少 GPU HBM 读写和
attention matrix materialization，适合电脑端训练。Nspire 的逐 token C decode
已经使用 KV cache，硬件和 kernel 完全不同；把训练端换成 FlashAttention 不会
自动减少计算器上的 KV bytes。

## 10. 部署边界与下一课

本课所有 512 结果都只在 PyTorch/CUDA 上成立。当前
[`ng_model.c`](../../runtime/src/ng_model.c) 仍有 legacy
`block_size <= 128` loader gate，因此 512 checkpoint 尚未：

- 导出为新的 `.ngm`；
- 通过 Host C logits/sequence 对齐；
- 编译并运行 Ndless chat；
- 测量 CX II 真实峰值 RAM、TTFT 或 token/s。

这不是失败，而是清晰的阶段边界。下一步应先选择：

1. 在现有 MHA/learned-position 路线把 C gate、格式和 256/512 对齐，得到简单
   baseline；
2. 再以 GQA + ALiBi/RoPE 为新架构分支，比较相同 context、相同模型文件预算、
   相同 Nspire RAM 下的精度和速度；
3. CoT/RLVR 仍在 256 主架构上独立推进，避免把架构收益误写成 RL 收益。

Lesson 14 的结论不是“CoT 无用”，而是更精确的：

> 在当前 10.8M byte-level student、当前数据和 48/96 输出 token 预算下，
> SFT 能教会显式的 thinking/non-thinking 控制格式，但没有提高冻结题精度；
> 512 context 需要真实长序列续训，并带来线性 KV 内存成本。
