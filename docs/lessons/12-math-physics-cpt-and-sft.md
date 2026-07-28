# Lesson 12：数学物理 CPT、角色 SFT 与“会回答但不会算”

Lesson 11 的模型已经会一点英语 next-byte continuation，但它既没有见过真正的
角色 token，也没有学过“用户提问后应该回答”。本课完成两次严格分开的训练：

```text
English Base
  -> math/physics continued pretraining (CPT)
  -> role-aware supervised fine-tuning (SFT)
```

最终结果非常适合作为一课：

```text
SFT 格式有效率：0% -> 95.3%
SFT 精确任务正确率：0% -> 1.56%
```

模型明显学会了回答格式、结束位置和数学物理措辞，却仍几乎不会泛化计算。
这不是互相矛盾的结论，而是 **语言建模、指令格式和可验证推理是不同能力** 的
直接证据。

## 1. 本课完成了什么

新增的主要实现：

- [`lesson12_curriculum.py`](../../training/nanogpt_nspire/lesson12_curriculum.py)：
  严格 GSM8K/OASST1 parser 和可精确复算的物理 curriculum；
- [`lesson12_data.py`](../../training/nanogpt_nspire/lesson12_data.py)：
  pinned inputs、CPT replay、role-aware SFT shards 和冻结 evaluation；
- [`stage_train.py`](../../training/nanogpt_nspire/stage_train.py)：
  带完整 parent lineage 的 CPT/SFT trainer；
- [`assistant_eval.py`](../../training/nanogpt_nspire/assistant_eval.py)：
  真正的角色 prompt、greedy generation、EOS/role leak/数值/单位评分；
- [`base_train.py`](../../training/nanogpt_nspire/base_train.py)：
  修复 assistant-only mask 下的 eligible-window sampling。

机器可读结果：

- [`lesson12-data.json`](../../experiments/lesson12-data.json)
- [`lesson12-training.json`](../../experiments/lesson12-training.json)

完整实施计划：

- [`2026-07-28-lesson-12-math-physics-sft.md`](../plans/2026-07-28-lesson-12-math-physics-sft.md)

下载的公开数据、完整 evaluation records、run logs 和 checkpoint 仍只保存在
ignored `artifacts/`，不会进入 Git。

## 2. CPT 和 SFT 到底有什么区别

### 2.1 Continued pretraining

CPT 仍然做普通 causal next-token prediction。一个领域记录可能是：

```text
<BOS>
Physics problem: A system has mass 12 kg ...
Solution: Use F = m a. Substitution gives 84 N.
<EOS>
```

它没有 `<USER>` / `<ASSISTANT>` 对话边界。目标是让 Base 更熟悉：

- 数字、运算符和短公式；
- 数学题与解答的常见语言；
- 力、速度、功率、压强等基础物理量；
- 单位和结果的局部结构。

### 2.2 Supervised fine-tuning

SFT 把同类知识改写成真正的对话序列：

```text
<BOS>
<USER>Calculate 12 * 7.
<ASSISTANT>The answer is 84.
<EOS>
```

其 mask 是：

```text
token: BOS USER question... ASSISTANT answer... EOS
mask:   0    0      0...        0       1...    1
```

因此模型会看到用户问题，却不会因“复述用户问题”获得 loss 奖励。只有 assistant
answer 和最终 `<EOS>` 计入 SFT loss。

这次 UI 的 `[USER]` / `[AI]` 不再只是 metadata。模型实际输入中真的包含固定
ID：

```text
<USER>      258
<ASSISTANT> 259
```

## 3. 为什么必须冻结 parent checkpoint

如果 CPT 或 SFT 偷偷从随机参数重新开始，就无法回答“后训练增加了什么”。本课的
lineage 是硬约束：

```text
CPT parent route = English-Base-Pilot
SFT parent route = Math-Physics-CPT
```

加载 parent 时逐项检查：

- checkpoint SHA-256；
- route 与 schema version；
- `vocab_size = 264` 和 tokenizer kind；
- 完整模型配置；
- state-dict key；
- 每个 tensor 的 shape、dtype 和有限性。

任何一项不一致都会在 optimizer 创建前失败。两个正式 checkpoint 的链为：

```text
Base
dcf6d3b4...de8be2
  |
  v
CPT
ab17a536...f89dfb
  |
  v
SFT
a65faad9...36a12
```

三者完全使用同一个 `6 layers / 6 heads / width 384 / context 256`
架构，变化来自数据与训练目标，不来自增大模型。

## 4. 数据来源与许可边界

### 4.1 GSM8K

官方 [GSM8K repository](https://github.com/openai/grade-school-math)
固定在 commit：

```text
3101c7d5072418e28b9008a6636bde82a006892c
```

原始数据共有：

| 原 split | 总行数 | 可放入 256-byte 完整问答 |
|---|---:|---:|
| train | 7,473 | 4,337 |
| test | 1,319 | 732 |

被拒绝的记录全是 `question + concise answer + role tokens > 256`。我们没有截断
问题或答案来假装它仍是完整训练样本。

原始 test 永远不进入 CPT/SFT shard，只用于 evaluation。最终答案只从官方
`#### number` 尾行解析，不执行 GSM8K 中的 calculation annotation。

### 4.2 OASST1

[OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) 固定在：

```text
fdf72ae0827c1cda404aff25b6603abec9e3399b
```

88,838 个 message 经过以下门：

- prompt 与 answer 都是 English；
- 非 synthetic、非 deleted；
- tree 已 `ready_for_export`；
- review 通过；
- assistant `rank = 0`；
- quality 至少 `0.5`；
- toxicity 不高于 `0.5`；
- root prompt 到 direct reply；
- 完整对话不超过 256 byte tokens。

最终只剩 255 对。小模型的上下文很短，所以宁可留下少量完整记录，也不把长回答
从中间切断后当成“高质量结束”。

### 4.3 项目生成算术与物理

项目生成：

- 12,000 个 integer/decimal/parenthesized arithmetic family；
- 4,000 个 introductory-physics numerical family；
- 10 种公式：
  density、force、kinetic energy、momentum、Ohm's law、power、pressure、
  speed、wave speed、weight。

所有数值使用 `Decimal` 或已有 `Fraction` 逻辑重新计算，不使用 Python
`eval`。物理记录同时保存 formula、quantity、unit 和 exact answer。

它们适合建立可审计的窄任务基线，但不是完整物理教材。概念性物理解释和更广
覆盖仍留给后续经过 verifier 的 teacher 数据。

Lesson 10 registry 中的 DeepMind Mathematics 仍是 eligible source，但本次
pilot 没有把它再混入。原因是它与项目生成的 programmatic school arithmetic
高度重叠，而本课已经同时有 exact generated curriculum 和 human-written
GSM8K；再加入第四个数学来源会让首次 CPT/SFT 的收益更难归因。它没有被排除，
只是保留给后续独立数据消融或 evaluation。

## 5. 为什么另外增加 easy arithmetic holdout

原 arithmetic generator 包含负数、小数、大乘法和括号。只从这个池随机抽
32 题，会回答“混合难度算术如何”，却不能单独回答：

```text
2 + 3
6 * 2
12 / 3
```

所以本课又枚举 `0..20` 四则运算，并执行两道隔离：

1. 排除所有已经进入 12,000-family 训练池的 family；
2. 只保留 family hash 落在固定 test bucket 的记录。

最终有 73 个 easy holdout，本次正式 generation 从中固定选择 32 个。不能为
凑到 256 而复制题目或放宽 test split。

## 6. 两份训练 shard

### 6.1 CPT

CPT 把 Lesson 11 原始 general-English shards 与新 domain shards 直接按 split
拼接。拼接前会重新验证每个 component manifest 的 byte count 与 SHA-256。

| 项目 | 观察值 |
|---|---:|
| domain CPT records | 20,337 |
| domain CPT tokens | 2,406,210 |
| composite CPT tokens | 6,718,812 |
| composite train tokens | 6,083,582 |
| general replay train fraction | 64.26% |

64.26% 是按实际 token 计算，不是按 document 数猜测。它用于降低领域继续训练
对通用英语的遗忘。

### 6.2 SFT

| 项目 | 观察值 |
|---|---:|
| SFT records | 36,592 |
| SFT families | 20,592 |
| SFT total tokens | 2,793,452 |
| SFT train tokens | 2,525,068 |
| SFT train assistant targets | 789,938 |

一个 family 可以有 direct 和 worked 两种回答，但它们永远进入同一个 split。
family 数小于 record 数正是这种设计的结果。

两次完整 build 的 root manifest SHA-256 都是：

```text
e337df15d070c0000d1ba88c4ec62b60fb561818cab8c87224ae5e3adb92d88e
```

## 7. SFT 暴露了 sampler 的一个隐藏假设

Base shard 几乎每个普通 byte 都是 eligible target，因此旧 sampler 隐含认为：

```text
随机抽一个 block -> block 内一定存在 loss target
```

SFT 不成立。一个 block 可能完全落在长 USER question 中：

```text
mask = 0 0 0 0 ... 0
```

正确修复不是把 USER mask 改成 1，而是：

- 随机训练时，用同一个 seeded generator 重采样，直到窗口至少有一个
  assistant target；
- full sequential evaluation 仍覆盖所有 prediction position；
- 如果某个 evaluation window 没有 eligible target，只计覆盖位置，不伪造
  loss。

这样既保持 deterministic sampling，也保持 assistant-only 目标不变。

## 8. 正式训练设置

两段都在同一 RTX 5080 Laptop GPU 上运行：

| 项目 | CPT | SFT |
|---|---:|---:|
| optimizer updates | 1,000 | 1,000 |
| tokens/update | 4,096 | 4,096 |
| 总 sampled tokens | 4,096,000 | 4,096,000 |
| max/min LR | `3e-4 / 3e-5` | `1e-4 / 1e-5` |
| warmup | 50 | 50 |
| BF16 autocast | 是 | 是 |
| peak CUDA allocation | 393,493,504 B | 393,493,504 B |
| optimizer throughput | 42,783 token/s | 27,251 token/s |
| loop wall time | 97.4 s | 152.8 s |

SFT 的 token/s 更低，部分原因是 sampler 需要避开纯 USER window；更重要的是，
“总 token”与“真正计入 loss 的 assistant target”不同：

```text
CPT sampled eligible targets: 4,083,096
SFT sampled eligible targets: 1,288,536
```

所以 SFT 大约覆盖了 `1.63` 个 assistant-target epoch，而不是把
`4,096,000 / 789,938` 直接当成有效 epoch。

## 9. Training loss 表明了什么

| Stage | initial sampled val | selected sampled val | full val | full test |
|---|---:|---:|---:|---:|
| CPT | 2.1697 | 1.4695 | 1.5287 | 1.4333 |
| SFT | 1.3664 | 0.3992 | 0.4683 | 0.4632 |

两段最佳点都在 step 1000。它说明在当前预算内 validation loss 尚未回升，不说明
继续无限训练一定更好，也不说明精确答案已正确。

sampled validation 与 full validation 不同，因为前者用固定随机窗口快速选点，
后者逐位置覆盖整个 split。checkpoint 只按 validation 选择，test 不参与选择。

## 10. 交叉 loss：CPT、SFT 与遗忘

将三个 checkpoint 放到三份相同 validation corpus 上：

| Checkpoint | general English | domain CPT | role SFT |
|---|---:|---:|---:|
| Base | 2.1200 | 2.4197 | 2.8036 |
| CPT | **1.9212** | **0.7892** | 1.3884 |
| SFT | 2.1145 | 1.1830 | **0.4683** |

可以分三步读：

1. CPT 显著改善 domain loss；
2. 因为有 64.26% general replay，而且总训练继续增加，CPT 的 general-English
   loss 也优于 Base；
3. SFT 强烈改善 assistant-only loss，但 general/domain loss 相对 CPT 回升。

SFT 的 general loss `2.1145` 仍略优于原 Base `2.1200`，所以不是“英语完全
坏掉”；但它确实失去了 CPT checkpoint 的一部分通用 continuation 优势。这就是
可测量的 specialization/catastrophic-forgetting trade-off。

## 11. 真正 generation 的公平比较

三个 checkpoint 都使用：

```text
<BOS><USER>prompt<ASSISTANT>
greedy decoding
最多 48 个新 byte token
遇到 <EOS> 停止
每类固定 32 个 family
```

评分同时检查：

- 最后一个可解析 decimal；
- physics unit；
- `<EOS>` termination；
- 是否泄漏其他 special token；
- 是否出现重复三词循环；
- 全部条件都通过后的 task correctness。

结果：

| Checkpoint | format valid | exact task accuracy | repeated phrase |
|---|---:|---:|---:|
| Base | 0.0% | 0.0% | 21.9% |
| CPT | 47.7% | 0.0% | 0.8% |
| SFT | **95.3%** | **1.56%** | **0.0%** |

SFT 分任务：

| Task | exact accuracy | termination |
|---|---:|---:|
| mixed arithmetic | 0/32 | 32/32 |
| easy arithmetic | 1/32 | 32/32 |
| GSM8K | 0/32 | 32/32 |
| physics numeric | 1/32 | 31/32 |

没有 checkpoint 在这 128 题中泄漏 role special token。

## 12. 两个成功例子为什么仍不能证明“会算”

### 12.1 `6 * 2`

```text
Base: м 1 1 1 1 ...                    [no EOS]
CPT:  Substitution gives 10 N.          [wrong]
SFT:  The answer is 12.                 [correct]
```

### 12.2 `P = E / t`

Prompt：

```text
A system has energy 300 J and time 15 s. What is its power?
```

生成：

```text
Base: The cone the state the seare ...  [wrong, no EOS]
CPT:  What is its power? Solution: ...  [wrong, no EOS]
SFT:  The power is 20 W.                [correct]
```

但 `1/32` 不是可靠能力。模型在其他 easy arithmetic 上经常输出 `10`、`11`
或 `12` 吸引值，例如：

```text
8 + 2  -> The answer is 12.
4 + 4  -> The answer is 12.
2 * 9  -> The answer is 12.
```

因此这里的正确例子可能包含局部模式或偶然命中，不能被挑出来当成整体能力证明。

## 13. 为什么 loss 低而 exact accuracy 仍低

teacher-forced SFT loss 计算：

```text
给定 gold answer 的前缀，预测下一个 gold byte
```

它可以因为大量固定前缀学得很好：

```text
The answer is
The force is
The power is
```

数字部分即使只错一个 byte，平均 loss 可能仍很低，但 exact-match 已经整题失败。
greedy generation 还会把前面自己的错误继续作为后续输入，这与 teacher forcing
不同。

所以：

```text
low assistant-token loss
!= exact arithmetic
!= multi-step reasoning
```

这也是为什么后续必须保留 verifier-based exact accuracy，而不能只看漂亮文本或
validation loss。

## 14. 这是不是一个真正的聊天模型了

现在可以更精确地回答：

- 它仍是一个真实的 decoder-only GPT；
- 它已经接受真实 role-aware SFT；
- `<USER>` / `<ASSISTANT>` 不再是 UI 装饰；
- 它能稳定形成短 assistant answer 并输出 `<EOS>`；
- 它是第一个真正的窄 instruction/chat baseline。

但它还不是可靠的数学物理助手。当前更准确的称呼是：

```text
role-aware tiny instruction model with weak task competence
```

## 15. 量化为什么还没有做

本课 checkpoint 是 FP32 training artifact，约 `43.7 MB`。架构与 Lesson 11
Student 完全相同，因此原 W4 静态估算仍约 `6.17 MB`，但这不是实际导出结果。

现在立即量化只能回答：

```text
一个几乎不会算的 SFT 模型，量化后还剩多少格式能力？
```

它不能解决能力不足。项目路线仍是：

```text
先改进 Base/SFT/teacher/RLVR 能力
-> 冻结最终候选
-> 真实 W4A8 export
-> PyTorch/Host C 对齐
-> Nspire 真机
```

量化不会被遗漏，只是不能把“文件变小”误写成“模型变聪明”。

## 16. 本课能宣称与不能宣称的事

现在有证据支持：

- pinned GSM8K/OASST1 与项目生成数据可重复构建；
- 原 GSM8K test 未进入训练；
- CPT 中 general replay 的实际比例已测量；
- SFT 的 USER/ASSISTANT tokens 与 assistant-only mask 正确；
- 纯 USER window sampler 问题已修复；
- parent checkpoint lineage 严格验证；
- CPT、SFT 在 CUDA 上完成并各自降低目标 validation loss；
- SFT 学会了短回答格式与 EOS；
- Base/CPT/SFT 的 loss、格式和 exact accuracy 已分别测量。

仍不能宣称：

- 纯神经模型已可靠掌握 `0..20` 算术；
- 模型能解决 GSM8K；
- 模型能可靠解释或计算基础物理；
- 低 SFT loss 等于推理能力；
- teacher、蒸馏、RLVR 或 CoT 已带来收益；
- W4A8 质量、Host C 对齐或 Nspire 性能已通过。

## 17. 下一课

Lesson 13 进入 teacher，但继续保持两种 teacher 不混淆：

1. external teacher 生成、批改短数学物理 sequence target；
2. local shared-tokenizer teacher 提供逐 token logits；
3. 从同一个 CPT/SFT lineage 比较 ordinary SFT、sequence distillation、
   logit distillation 和组合路线；
4. 固定本课的 128-prompt generation gate，不事后换题；
5. 仍把 exact verifier 结果与格式/loss 分开报告。

外部 API key 必须先轮换，并且只从环境变量读取。Lesson 13 也不会把 teacher
答案直接当 gold：数值、单位和 family leakage 仍要通过本地 verifier。

只有得到更强而且可验证的 SFT/distilled checkpoint 后，才进入同输出 token
限制下的 direct-answer/CoT RLVR，以及最终 W4A8/C/Nspire 部署。
