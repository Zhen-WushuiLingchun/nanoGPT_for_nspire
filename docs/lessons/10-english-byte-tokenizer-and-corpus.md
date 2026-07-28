# Lesson 10：从 Shakespeare 字符续写到英语 Base 数据地基

前九课已经证明：我们可以训练一个真正的 GPT，把权重量化成 W4A8，在 C 中与
PyTorch 对齐，并让它在 Nspire 上逐 token 生成。

但 Lesson 09 的真机照片也暴露了能力边界：

```text
USER: ONE PLUS TWO
AI:   ... THE STATE ...
```

这不是 UI 或 C runtime 在伪造答案。模型真的在执行它学到的任务：

```text
根据 Tiny Shakespeare 的前文，预测下一个字符
```

如果想得到一个英文数学物理助手，不能只修改 prompt。我们需要依次改变：

```text
tokenizer
训练语料
对话格式
训练目标
teacher
评测与可验证奖励
```

本课完成前三项所需的数据地基，但**还没有训练新的 Base checkpoint**。这一边界
很重要：数据文件能成功生成，不等于模型已经会英语、数学或物理。

## 1. 这一次完成了什么

代码新增：

- [`byte_tokenizer.py`](../../training/nanogpt_nspire/byte_tokenizer.py)：
  256 byte token 与 8 个固定 special token；
- [`source_registry.py`](../../training/nanogpt_nspire/source_registry.py)：
  公开数据来源、版本、许可和排除理由；
- [`math_curriculum.py`](../../training/nanogpt_nspire/math_curriculum.py)：
  不使用 `eval` 的精确算术生成器；
- [`base_corpus.py`](../../training/nanogpt_nspire/base_corpus.py)：
  family-level split、去重、编码、loss mask 与原子 shard；
- [`lesson10_data.py`](../../training/nanogpt_nspire/lesson10_data.py)：
  可重复的 Lesson 10 smoke 命令。

长期设计与逐步计划分别位于：

- [`english-math-physics-assistant-design.md`](../plans/2026-07-28-english-math-physics-assistant-design.md)
- [`lesson-10-english-base-data.md`](../plans/2026-07-28-lesson-10-english-base-data.md)

## 2. 为什么不继续使用 65 字符词表

Tiny Shakespeare 的词表来自训练文本中出现过的字符：

```text
vocab_size = 65
```

这种做法适合第一课，因为每个 token 都能直接看懂。但它有三个限制：

1. 没出现过的字符无法编码；
2. 换数据集就可能改变 `vocab_size` 和模型 tensor shape；
3. `USER`、`ASSISTANT`、`TOOL` 等结构没有稳定 token ID。

本课冻结的新词表为：

```text
0..255  = raw byte
256     = <BOS>
257     = <EOS>
258     = <USER>
259     = <ASSISTANT>
260     = <TOOL>
261     = <THINK>
262     = <FINAL>
263     = <PAD>

vocab_size = 264
```

所有 UTF-8 文本都能先变成 bytes：

```text
text
  -> UTF-8 encode
  -> byte 0..255
  -> token ID 0..255
```

因此不再需要 `<UNK>`。英文字母和普通 ASCII 数学符号通常是一 byte；`Δ` 等
Unicode 符号会变成多个 byte token。

测试把：

```python
bytes(range(256))
```

全部编码再解码，要求逐 byte 相等。special token 不能被 `decode_bytes` 误当成
文本；调试显示会明确写成 `<USER>`，而不是把它伪装成若干普通字符。

## 3. character、byte 与 BPE

| tokenizer | 优点 | 缺点 | 本阶段选择 |
|---|---|---|---|
| 数据集字符词表 | 最直观、词表小 | 换语料易 OOV | 淘汰 |
| byte | 固定、无 OOV、C 实现简单 | 英文序列比 BPE 长 | **采用** |
| BPE/subword | 英文压缩率高 | 训练/编码/C 移植更复杂 | 后续对照 |

byte tokenizer 并不会自动让模型变聪明。它只是让数据边界稳定，并允许我们在
训练前冻结 embedding 和 vocabulary head 的形状。

对于 Nspire，byte 还有一个工程优势：C 端查表和输出可以按 raw byte 实现，不必
先移植完整 BPE merge 算法。但 byte 序列更长，所以初版 Base 的 context 从
`128` 计划提高到 `256`；是否继续到 `384/512` 必须经过文件、KV RAM、TTFT 和
真机速度测量。

## 4. USER/ASSISTANT 终于成为模型 token

Lesson 09 的角色只存在于 UI：

```text
[USER]  <- cell metadata
[AI]    <- cell metadata
```

模型实际只看到了用户字符和 continuation。现在单轮 SFT record 会编码为：

```text
<BOS>
<USER>
What is 12 * 7?
<ASSISTANT>
12 times 7 is 84.
<EOS>
```

对应 token 流：

```text
256
258
question UTF-8 bytes
259
answer UTF-8 bytes
257
```

后续轮次重复 `<USER>...<ASSISTANT>...`。这意味着角色边界是模型输入的一部分，
不是界面装饰。

注意：special token 本身也没有魔法。只有在大量正确格式的 SFT example 中，
模型才会学到：

```text
<USER> 后面通常是问题
<ASSISTANT> 后面应当生成回答
```

## 5. 为什么只让 assistant answer 产生 SFT loss

对于：

```text
<BOS><USER>Q<ASSISTANT>A<EOS>
```

本课生成同长度的 loss mask：

```text
token       BOS USER Q... ASSISTANT A... EOS
loss mask    0   0   0...     0      1...  1
```

用户问题的 token 仍然作为上下文输入，但不要求模型“背诵用户说了什么”。训练
重点变成在给定问题与角色边界后预测 assistant answer。

Base 预训练不是这样。普通文档编码为：

```text
<BOS> document bytes <EOS>
```

除 `<BOS>` 外，文档与 `<EOS>` 都是 next-token target。于是两个阶段仍然共享
同一个基本目标：

```text
cross entropy(next token)
```

区别在于 SFT 使用 mask 选择哪些 target 计入平均 loss。

## 6. “真正的 Base 模型”仍然只是 next-token model

Base training 不使用 USER/ASSISTANT 问答作为主要数据，而是学习连续英文：

```text
educational prose
mathematical prose
short explanations
symbols and units
```

它首先需要学会英文句法、常见词序、数字和基本概念。之后才做：

```text
Base -> continued pretraining -> SFT
```

直接从随机参数对少量问答做 SFT，模型很容易只记住模板，或者输出看似对话但
语言分布极差。RLVR 也不能从零补上英语能力：reward 可以告诉模型最终答案是否
正确，却不会自动提供完整语言模型需要的分布。

## 7. 公开数据集如何选择

机器可读表位于
[`lesson10-public-sources.json`](../../experiments/lesson10-public-sources.json)。

首批计划使用：

| 来源 | 用途 | 许可/规则 |
|---|---|---|
| FineWeb-Edu | 高分英语教育文本 | ODC-By-1.0 |
| Common Corpus | 可追踪开放英文文档 | 每行只接受 Public Domain、CC0、CC-BY |
| OpenWebMath | 数学继续预训练 | ODC-By-1.0 |
| DeepMind Mathematics | 程序化数学 curriculum | Apache-2.0 |
| GSM8K | 人工数学文字题 | MIT repo；原 test 保持隔离 |
| OpenMathInstruct-2 | 短、可验证数学解答 | CC-BY-4.0 |
| OASST1 | 少量英语对话格式 | Apache-2.0 |
| DeepSeek V4-Pro generated | 物理解释与 sequence distillation | 保存 AI/provenance/verifier 标记 |

数据公开不等于可以随意训练。首版 permissive mix 默认拒绝：

```text
unknown license
CC-BY-NC
CC-BY-SA
AI-training prohibited
```

OpenStax College Physics 2e 当前页面明确写有未经许可不得将内容摄入 LLM 或
生成式 AI，因此即使页面同时显示 CC BY-NC-SA，本项目也不使用它。SciQ 是
CC-BY-NC-3.0；OpenBookQA 的公开 dataset card 未给出确定许可。三者都以明确
排除理由留在 registry 中，而不是从表里消失。

为什么还要记录“排除项”？因为未来换数据的人会看到：

```text
这个来源是没发现，还是审查后有意不用？
```

## 8. 数据质量不是“下载成功”

一条 record 进入训练前要通过：

```text
source exists
  -> policy/license eligible
  -> strict UTF-8
  -> non-empty and bounded
  -> normalized fingerprint
  -> duplicate check
  -> family split
  -> task verifier
  -> token encoding
```

下载 HTTP 200、JSON 能解析或 teacher 说得流畅，都不等于高质量。

对于公开大语料，后续还要增加：

- English language score；
- educational/math score；
- 文档长度和格式过滤；
- boilerplate 与控制字符过滤；
- exact/near duplicate；
- benchmark contamination；
- source revision 和 per-row attribution。

Lesson 10 暂时没有下载 FineWeb-Edu 等大语料，也没有调用 DeepSeek。它先保证
数据合同不会在下载之后再临时改变。

## 9. 为什么按 family split，而不是按 record 随机 split

假设一个算术 seed 产生：

```text
Calculate 12 * 7.
What is twelve times seven?
Give a worked solution for 12 * 7.
```

它们不是三个独立问题，而是同一个 family 的变体。如果逐 record 随机分割，很
可能出现：

```text
train:  Calculate 12 * 7.
test:   What is twelve times seven?
```

测试准确率就会夸大泛化能力。

本课先计算：

```text
SHA256(split_seed + ":" + family_id)
```

再按固定 bucket 分成：

```text
train       90%
validation   5%
test         5%
```

同一 family 的 direct answer、worked answer、未来 CoT、paraphrase 和 tool form
永远得到相同 split。

split 必须发生在 teacher paraphrase 之前。否则 teacher 生成的相似问题可能各自
拥有不同 ID，污染已经发生后再去重会很困难。

## 10. 为什么不能用 Python eval 验证数学题

下面这种代码不安全：

```python
answer = eval(expression)
```

它把数据当成代码执行，而且浮点数还可能引入：

```text
0.1 + 0.2 != 0.3
```

本课只允许显式运算符：

```text
+  -  *  /
```

整数和除法使用 `Fraction`，小数使用 `Decimal`。表达式树由生成器直接构造，
verifier 用同一组显式 operator 重新计算。例如：

```text
(7 + 5) * 3

First, 7 + 5 = 12.
Then, 12 * 3 = 36.
```

如果有人只把保存的 `exact_answer` 从 `36` 改成 `37`，重新计算会失败。division
by zero、NaN、无穷、过长数字和未支持运算符都会在生成前被拒绝。

## 11. Lesson 10 smoke 实验

命令：

```powershell
python -m nanogpt_nspire.lesson10_data smoke `
  --output artifacts/lesson10-smoke `
  --seed 20260728 `
  --examples 256
```

每个 arithmetic family 生成：

```text
direct answer
worked answer
```

实际结果：

| 指标 | 结果 |
|---|---:|
| arithmetic families | 256 |
| records | 512 |
| train families / records | 227 / 454 |
| validation families / records | 14 / 28 |
| test families / records | 15 / 30 |
| total byte/special tokens | 22,804 |
| duplicates removed | 0 |
| vocabulary | 264 |

两次独立输出的全部文件 byte-identical，manifest SHA-256 都是：

```text
f4c5938fdc2d7f0fb501031e19c7e1daec987f956d2aac1cf0b70f175425629a
```

shard 采用：

```text
*.tokens.bin  little-endian uint16
*.loss.bin    uint8 0/1
```

manifest 保存每个文件的 byte count 与 SHA-256。生成目录在 `artifacts/`，不会
进入 Git；仓库只保存小型证据摘要
[`lesson10-base-data.json`](../../experiments/lesson10-base-data.json)。

## 12. 原子写入与 replace 边界

builder 先写临时 sibling directory：

```text
.lesson10-smoke.tmp-...
```

所有 shard、mask 和 manifest 完成后，才一次 rename 为正式目录。中途失败不会
留下一个看似完整的数据集。

已有输出默认拒绝覆盖。只有显式 `--replace` 才允许替换，而且目标必须先通过：

- 是真实 directory，不是 symlink；
- 存在可解析 manifest；
- corpus schema、tokenizer vocab 和 split kind 与 Lesson 10 一致。

随便创建一个含个人文件的目录，即使传 `--replace` 也不会删除。

## 13. DeepSeek Teacher 如何进入，而不污染密钥和 gold test

当前正式外部 teacher 计划使用：

```text
deepseek-v4-pro
```

旧 `deepseek-chat` / `deepseek-reasoner` 名称不写进新管线。API key 只从：

```text
DEEPSEEK_API_KEY
```

读取，不进入参数、异常、manifest、prompt record 或 Git。Lesson 10 没有调用 API。

后续数据生成流程是：

```text
permissive seed fact / project formula
  -> V4-Pro candidate question + reasoning + final
  -> exact math or unit verifier
  -> independent critique pass
  -> accept or quarantine
```

冻结的 held-out test 不发送给 teacher。否则 teacher 已经见过测试题，我们就无法
区分 student 泛化和 sequence memorization。

## 14. CoT 与 RLVR 如何做公平比较

我们会从同一个 Base/SFT 起点派生：

```text
Direct-answer SFT
CoT-SFT
Direct-answer RLVR
CoT-RLVR
```

CoT 格式：

```text
<ASSISTANT><THINK>reasoning<FINAL>answer<EOS>
```

所有路线共享：

- architecture；
- tokenizer；
- evaluation questions；
- maximum generated tokens；
- decoding policy；
- final-answer parser。

reasoning token 也计入输出上限。即使 UI 不显示 `<THINK>`，它仍消耗：

```text
context
decode time
energy
token budget
```

因此报告至少包含：

- final-answer accuracy；
- invalid-format rate；
- reasoning/output token count；
- TTFT；
- tokens/s；
- 达到 token 上限的比例。

RLVR reward 只根据可解析的最终答案、精确数值或容差、以及单位检查计算。不能因为
推理看起来流畅就奖励错误答案。

## 15. 纯模型和 calculator tool 仍然是两条路线

纯模型：

```text
question -> GPT tokens -> answer
```

tool-assisted：

```text
question
  -> expression detection
  -> safe C arithmetic parser
  -> exact result
  -> model formats explanation
```

工具路线能保证 `12 * 7 = 84`，但它不能用来声称纯神经模型学会了乘法。二者必须
分别评分。`<TOOL>` 已预留 token ID，但本课没有实现 C parser 或 tool-use SFT。

## 16. 验证结果与尚未成立的结论

本课回归：

```text
Python: 254 passed
CTest:  7 / 7 passed
```

已经成立：

- 256 bytes 全量 round-trip；
- 8 个 special token ID 已冻结；
- USER/ASSISTANT 对话与 assistant-only loss mask 可重复；
- 未知/NC/SA/AI 禁止来源默认被拒绝；
- 算术生成与 verifier 不使用 `eval`；
- family variants 不跨 split；
- 两次 smoke build 全文件 byte-identical；
- Lesson 09 C runtime/UI 未因数据层变化而回归。

尚未成立：

- FineWeb-Edu/Common Corpus/OpenWebMath 已完成正式下载与质量抽样；
- 新英语 Base 已训练；
- Base 会稳定生成英文；
- SFT 后能正确回答数学或物理；
- DeepSeek 数据已经生成或验证；
- CoT/RLVR 有收益；
- 新 byte/special-token 模型能由 `.ngm` 和 C runtime 加载；
- 新模型已重新量化或部署到 Nspire。

下一课将先解决 CUDA PyTorch 环境，冻结 student/teacher 的参数、文件和 KV RAM
估算，再进行小规模 Base pilot。只有 pilot 的 train/validation loss、固定英文
completion 和数据吞吐通过门槛，才扩大到约 256M byte-token 的正式 student
pretraining。
