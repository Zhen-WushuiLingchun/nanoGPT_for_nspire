# Lesson 11：第一次训练真正的英语 Base 模型

Lesson 10 只完成了 tokenizer、语料格式和许可地基。本课第一次把随机初始化的
`6×384` GPT 放到真实英文教育语料上，执行 causal next-token pretraining。

结果可以用两句话概括：

```text
完整 validation loss: 5.733 -> 2.120
固定 prompt 生成：像英语，但仍没有可靠语义
```

这两句话并不矛盾。第一句证明模型学会了语料中的 byte 序列规律；第二句提醒我们：
Base pretraining、指令跟随和正确回答问题是不同能力，不能用一个 loss 数字混为
一谈。

## 1. 本课完成了什么

新增的主要代码是：

- [`cuda_probe.py`](../../training/nanogpt_nspire/cuda_probe.py)：独立 CUDA
  环境与真实矩阵 workload；
- [`model_budget.py`](../../training/nanogpt_nspire/model_budget.py)：参数、
  W4 文件和 C inference arena 静态账本；
- [`public_corpus.py`](../../training/nanogpt_nspire/public_corpus.py)：固定
  Parquet commit、质量门、hash selection 与逐文档 provenance；
- [`lesson11_data.py`](../../training/nanogpt_nspire/lesson11_data.py)：
  FineWeb-Edu/OpenWebMath pilot 命令；
- [`base_train.py`](../../training/nanogpt_nspire/base_train.py)：memory-mapped
  byte shard、masked causal loss、BF16、梯度累积、验证和 best checkpoint。

机器可读证据分为四份：

- [`lesson11-cuda-environment.json`](../../experiments/lesson11-cuda-environment.json)
- [`lesson11-model-budget.json`](../../experiments/lesson11-model-budget.json)
- [`lesson11-public-pilot-data.json`](../../experiments/lesson11-public-pilot-data.json)
- [`lesson11-base-pilot.json`](../../experiments/lesson11-base-pilot.json)

实现计划位于
[`2026-07-28-lesson-11-english-base-pilot.md`](../plans/2026-07-28-lesson-11-english-base-pilot.md)。

## 2. 它现在为什么是“真正的 Base 语言模型”

一个 decoder-only Base 模型的核心任务仍然非常朴素：

```text
给定前面的 token，预测下一个 token
```

例如一篇文档编码成：

```text
<BOS> F o r c e ... . <EOS>
```

训练窗口取：

```text
inputs  = tokens[s : s + T]
targets = tokens[s + 1 : s + T + 1]
```

因此每个 target 都向左移动一格。causal mask 保证位置 `t` 只能读取
`0..t`，不能偷看更右边的答案。

本课不是把固定句子写进 UI，也不是调用云端模型。checkpoint 从随机参数出发，
通过 PyTorch 的 forward、cross entropy、backward 和 AdamW 更新得到。它就是
一个真实的小型 decoder-only Transformer。

但“真实 LLM”不等于“已经是 ChatGPT”。当前 checkpoint：

- 学过英文教育文档的 next-byte distribution；
- 没见过 `<USER>...<ASSISTANT>...` 指令训练样本；
- 没有经过数学答案 verifier；
- 没有经过 teacher 蒸馏；
- 没有偏好优化或 RLVR。

所以它是 **Base LM**，还不是 **instruction/chat model**。

## 3. 先解决 RTX 5080 的 CUDA 环境

主仓库原来的 `.venv` 是 CPU PyTorch。本课没有覆盖或降级它，而是在 Lesson 11
worktree 中创建独立 `.venv`：

```powershell
uv venv --python 3.12 .venv

uv pip install --python .venv\Scripts\python.exe `
  torch==2.11.0 `
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv\Scripts\python.exe `
  -e ".[test,lesson11]"
```

实测环境：

| 项目 | 观察值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| compute capability | 12.0 |
| driver | 591.74 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| Python | 3.12.12 |

探针执行 5 次 `4096×4096` FP32 matrix multiply + reduce，得到有限 checksum，
峰值 PyTorch allocation 为 `276,957,696 bytes`。这个探针只回答：

```text
CUDA wheel 能否在这张 GPU 上真正计算？
```

它不是 GPT 训练 benchmark，也不能替代后面的 tokens/s 和 peak VRAM 记录。

PyTorch 2.7 起的 CUDA 12.8 wheel 已加入 Blackwell 支持；本课使用更新的
2.11 CUDA 12.8 wheel。安装与验证原则来自
[PyTorch Get Started](https://pytorch.org/get-started/locally/) 和
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)。

## 4. 为什么要在训练前做模型预算

电脑能训练，不代表计算器能部署。Student 的最终权重必须进入约 4–6 MiB 文件，
而计算器推理峰值 RAM 不能超过 24 MiB。

本课比较了四个 student 候选：

| 候选 | 参数量 | 估计 W4 文件 | 估计峰值 RAM | 文件门 |
|---|---:|---:|---:|---|
| 6×320 | 7,543,360 | 4,326,208 B | 10,373,056 B | 通过 |
| 6×384 | 10,821,504 | 6,172,992 B | 13,009,344 B | 通过 |
| 8×320 | 10,002,240 | 5,714,496 B | 13,072,064 B | 通过 |
| 6×448 | 14,689,472 | 8,351,616 B | 15,977,536 B | 失败 |

最终冻结：

```text
Student
layers       6
heads        6
width        384
head_dim     64
context      256
vocab        264
parameters   10,821,504
```

`6×384` 比 `6×320` 和 `8×320` 使用更多参数，同时只有 6 个串行 block。
`n_embd / n_head = 384 / 6 = 64`，每个 head 的宽度也是规则整数。

### 4.1 参数量从哪里来

不带 bias、embedding/head tied 时：

```text
token embedding       vocab * C
position embedding    T * C

per block matrices    (4 + 2 * mlp_ratio) * C^2
per block norms       2 * C

final norm            C
```

`mlp_ratio=4`，所以每个 block 的 matrix 部分是：

```text
(4 + 2*4) * C^2 = 12 * C^2
```

把 `C=384`、`layers=6`、`vocab=264`、`T=256` 代入，得到
`10,821,504` 个唯一参数。单元测试同时实例化 tied 和 untied PyTorch 模型，要求
公式与 `sum(parameter.numel())` 完全相等。

### 4.2 为什么 10.8M 参数可能装进 6 MiB

FP32 原始参数需要：

```text
10,821,504 * 4 = 43,286,016 bytes
```

显然不能直接部署。W4 路线把二维 matrix 沿最后一维按 64 个值分组：

```text
matrix weights        signed INT4 packed
one-dimensional norm FP32 passthrough
each group scale      FP32
```

Student 的静态账本：

| 组成 | bytes |
|---|---:|
| packed INT4 matrix | 5,408,256 |
| FP32 group scales | 676,032 |
| FP32 1-D tensors | 19,968 |
| logical payload | 6,104,256 |
| header/table/tokenizer/alignment/v2 reserve 后 | 6,172,992 |
| 6 MiB limit | 6,291,456 |
| margin | 118,464 |

只有约 `116 KiB` 余量，所以这里必须写“估计通过”，不能写“已经部署成功”。
Lesson 15 的真实 `.ngm v2` 如果超过上限，就必须缩模型或修改量化格式，不能把
估算当成事实。

### 4.3 文件大小、训练显存、推理 RAM 是三回事

静态推理 RAM 还包括：

```text
model blob                         6,172,992
FP32 K/V cache                     4,718,592
float workspace                       19,072
quantized activation workspace        1,536
UI/allocator safety reserve        2,097,152
------------------------------------------------
estimated peak                    13,009,344
```

它低于 `25,165,824 bytes` 的 24 MiB 门。但这仍不是 Nspire 真机峰值测量。

训练时还要保存 gradient、Adam 的一阶/二阶矩、activation 和 kernel
temporary，因此训练 VRAM 与 W4 文件大小没有直接相等关系。本次 BF16 pilot 的
真实 PyTorch peak allocation 是 `393,493,504 bytes`。

### 4.4 Teacher 为什么不部署

本课同时冻结电脑端 Teacher：

```text
layers       12
heads        10
width        640
context      256
parameters   59,331,200
```

其估计 FP32 文件约 `237,395,840 bytes`，即使 W4 也约
`33,499,840 bytes`，都远超 6 MiB。因此 Teacher 的职责是提高 student 的训练
信号，而不是直接塞进计算器。

## 5. 真实公开语料是怎样选出来的

本课没有下载整个 FineWeb-Edu 或 OpenWebMath。两个输入都固定到 auto-converted
Parquet 的 40 位 commit：

| source | exact revision | 扫描 row group |
|---|---|---|
| FineWeb-Edu | `92cece42bcce787ee4af4619ab449fe48d86230d` | 0, 181, 363, 545 |
| OpenWebMath | `c5476cfea8186f9db20fe4b45f43fa2e231aa9ba` | 0, 14, 28, 42 |

通过 Hugging Face filesystem 和 PyArrow 的 HTTP range read，只读取指定 row
group；FineWeb 的完整 `0000.parquet` 约 2.15 GB，但本课没有把它完整下载。

数据卡分别是：

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math)

两者都按 ODC-By-1.0 登记；OpenWebMath 还要求遵守 Common Crawl 条款。模型以后
发布时必须继续保留适当 attribution 和数据 provenance。

### 5.1 FineWeb-Edu 质量门

扫描 4,000 rows，要求：

```text
language == "en"
language_score >= 0.9
int_score >= 4
valid UTF-8
256 <= normalized UTF-8 bytes <= 65,536
no forbidden control character
```

结果：

| 状态 | rows |
|---|---:|
| accepted | 481 |
| educational score < 4 | 3,000 |
| language confidence < 0.9 | 518 |
| Unicode replacement character | 1 |

原定 FineWeb byte cap 是 6 MiB，但全部合格文档只有 `2,213,788 bytes`。本课没有
为了凑数量而把 score 门降到 3。

### 5.2 OpenWebMath 质量门

同样扫描 4,000 rows。数据集本身已经执行 English/math/quality pipeline，本课
继续剔除太短、太长、replacement character 和控制字符文档，然后按固定 SHA-256
排名选到 2 MiB：

```text
accepted scan rows       3,824
selected documents         367
selected UTF-8 bytes  2,097,118
```

### 5.3 family split 和逐文档 provenance

最终语料：

| split | records | byte tokens |
|---|---:|---:|
| train | 770 | 3,909,439 |
| validation | 39 | 215,043 |
| test | 39 | 188,120 |
| total | 848 | 4,312,602 |

每篇文档独立成一个 `family_id`，使用 Lesson 10 的 SHA-256 90/5/5 split，所以同一
文档不会跨 split。

`provenance.jsonl` 不重复保存原文，但为每篇文档保存：

```text
source repository + exact revision + parquet path
row group + row index
source document ID + URL
license
normalized text byte count + SHA-256
quality evidence
```

完整构建重复两次，两个 outer manifest 的 SHA-256 都是：

```text
408b3eea27d4f3a734ca0c8cc458e8c621d69db7394675b02cdcf30086c4a8b8
```

并且所有 shard metadata 完全相同。

## 6. loss mask 为什么要向 targets 对齐

Lesson 10 的 base record loss mask 是：

```text
token       <BOS>  byte  byte ... <EOS>
loss mask      0     1     1  ...    1
```

训练时 target 向左移动，所以 mask 也必须取：

```python
inputs      = tokens[s : s + T]
targets     = tokens[s + 1 : s + T + 1]
target_mask = masks [s + 1 : s + T + 1]
```

于是模型：

- 学习正文 byte；
- 学习何时产生 `<EOS>`；
- 不要求预测下一篇文档的 `<BOS>`。

masked cross entropy 是：

```text
             Σ_i mask_i * [-log p(target_i | prefix)]
loss =       ------------------------------------------
                         Σ_i mask_i
```

如果错误地使用未 shift 的 mask，监督目标会错一位。这类 bug 不一定报错，却会
静默改变训练任务，所以测试直接检查 token、target 和 mask 的精确数组。

shard 使用 little-endian `uint16` memory map。4.31M token 不需要先复制成整块
PyTorch `int64`；只有采样到的 batch window 会转换并送进 GPU。

## 7. 一次 optimizer update 中发生了什么

冻结的训练 batch：

```text
micro_batch_size              4
gradient_accumulation_steps   4
context T                   256

effective tokens/update = 4 * 4 * 256 = 4,096
```

一次 update 的逻辑是：

```python
optimizer.zero_grad()

for micro_batch in range(4):
    logits = model(inputs)
    loss = masked_cross_entropy(logits, targets, mask)
    (loss / 4).backward()

clip_grad_norm_(..., 1.0)
optimizer.step()
```

梯度累积不等于 4 次 optimizer update。四个 micro-batch 的 gradient 先累加，
最后只更新一次参数，所以 effective batch 更大，而 activation peak 仍接近一个
micro-batch。

## 8. 完整 Student 的张量形状

本次训练：

```text
B = 4
T = 256
C = 384
H = 6
D = C/H = 64
V = 264
```

主要张量：

| 张量 | shape |
|---|---|
| token IDs | `(4, 256)` |
| token + position embedding | `(4, 256, 384)` |
| fused QKV | `(4, 256, 1152)` |
| Q/K/V after split | `(4, 6, 256, 64)` |
| attention scores | `(4, 6, 256, 256)` |
| attention output | `(4, 256, 384)` |
| MLP hidden | `(4, 256, 1536)` |
| logits | `(4, 256, 264)` |

梯度累积只重复这些 micro-batch shape，不把 batch 维直接扩成 16。

## 9. overfit gate：为什么 5 step 失败、20 step 通过

在正式遍历语料前，先固定一个 batch 反复训练。如果完整模型连同一小批数据都
学不会，通常说明 loss、mask、optimizer、AMP 或 backward 有问题。

第一次额外尝试只给 5 step，gate 失败。计划冻结的 20-step gate 则得到：

```text
initial repeated-batch loss   5.584885
final repeated-batch loss     2.797222
passed                        true
```

为什么 5 step 不是可靠门？

- 这是 10.8M 参数的随机模型；
- training mode 启用 `dropout=0.1`；
- 前 50 个正式 step 原本处于 learning-rate warmup；
- “一次 loss 抖动”与“系统无法学习”不是同一结论。

因此本课保留 5-step 失败现象，但采用预定的 20-step门。它不是把失败删掉，而是
说明 gate 的统计尺度必须与模型大小和随机正则化相匹配。

## 10. validation loss、byte perplexity 与 bits-per-byte

cross entropy 使用自然对数。两个常见变换是：

```text
byte perplexity = exp(loss)
bits per byte   = loss / ln(2)
```

注意这里是 **byte perplexity**，不能直接与使用 BPE tokenizer 的论文
perplexity 横向比较。

两个无 Transformer baseline：

| baseline | validation loss | 解释 |
|---|---:|---|
| uniform over 264 tokens | 5.575949 | 每个 token 概率相等 |
| add-one unigram frequency | 3.296311 | 只看全局频率，不看上下文 |

完整 token 流评估：

| 模型状态 | loss | byte perplexity | bits/byte |
|---|---:|---:|---:|
| random init validation | 5.733047 | 308.909 | 8.271 |
| selected validation | 2.119973 | 8.331 | 3.058 |
| selected test | 2.064447 | 7.881 | 2.978 |

selected model 显著优于 unigram baseline，说明它确实使用了上下文，而不只是学到
“空格和 e 很常见”。

### 10.1 sampled validation 与 full validation

训练中每 100 step 使用固定的 20 个 batch window，便于快速、可重复地选择
checkpoint。最后另外覆盖完整 validation/test target stream：

```text
sampled selected validation loss   2.140835
full selected validation loss      2.119973
full selected test loss            2.064447
```

二者接近，但职责不同：

- sampled validation：便宜的训练中监控；
- full validation/test：最终读数。

full evaluator 验证自己精确覆盖了 `215,004` 个 eligible validation targets 和
`188,081` 个 eligible test targets。

## 11. 训练曲线告诉了我们什么

训练参数：

```text
steps                  1,000
tokens/update          4,096
training tokens    4,096,000
approx epochs          1.048
AdamW max LR           6e-4
warmup steps              50
cosine min LR          6e-5
```

固定 sampled validation loss：

| step | loss |
|---:|---:|
| 0 | 5.728679 |
| 100 | 2.688700 |
| 200 | 2.652135 |
| 300 | 2.607887 |
| 400 | 2.564923 |
| 500 | 2.424696 |
| 600 | 2.310906 |
| 700 | 2.249562 |
| 800 | 2.194743 |
| 900 | 2.161827 |
| 1000 | 2.140835 |

best step 是最后的 1000，曲线仍在下降。因此这次没有看到“继续训练已经过拟合”的
证据。正确结论是：

```text
one-epoch pilot 成功，但尚未训练充分
```

不能因为最后一点仍下降，就无限重复这 4.31M token；更合理的下一步是扩大高质量
语料并加入数学/物理 continued pretraining，再监控 full validation。

## 12. 真实吞吐和显存

RTX 5080 Laptop 上：

| 指标 | 观察值 |
|---|---:|
| optimizer update time | 45.369 s |
| training wall time | 46.162 s |
| update throughput | 90,281 tokens/s |
| peak PyTorch CUDA allocation | 393,493,504 B |
| FP32 checkpoint | 43,705,491 B |

checkpoint 是训练用 FP32 PyTorch archive，不是 Nspire `.ngm`。它大于 6 MiB 是
正常的；Lesson 15 才会重新量化和导出。

同 seed 的两次 CUDA run 产生的 40 个 model-state tensor 全部 bit-identical。
两个 `.pt` 文件 hash 不同，因为 source commit metadata 不同；比较 checkpoint
容器 hash 不能替代逐 tensor 比较。

## 13. 为什么生成仍然像“英语乱码”

固定 prompts 的实际输出包括：

```text
Force is
 ->  the prober siatemer detround the be work sout ...

The value of x
 -> pencengen the can inference somefections ...

Energy can be
 ->  made to gisto the fally ...
```

这些输出已经学到：

- 空格、标点和换行位置；
- 英文常见字母组合；
- 类似词尾的局部模式；
- 某些训练语料的排版习惯。

但它没有稳定学到：

- 单词边界和完整词形；
- prompt 的问答意图；
- `force`、`x`、`energy` 的概念；
- 正确数学或物理推理。

原因不是 C runtime，也不是 UI role metadata，因为这些样本直接来自 PyTorch
checkpoint。主要限制是：

1. 只有约 4.31M byte tokens；
2. 仅训练约一个 epoch；
3. 10.8M 参数相对语料很大；
4. byte tokenizer 要先从 bytes 自己学出词形；
5. Base objective 只要求续写，不要求回答；
6. temperature/top-k sampling 会暴露分布尚不尖锐的问题。

这是本课最重要的现象之一：

```text
loss 大幅下降
≠ 已经有可用自然语言能力
≠ 已经会数学物理
≠ 已经是聊天助手
```

## 14. 如何复现实验

构建真实 public pilot：

```powershell
python -m nanogpt_nspire.lesson11_data `
  --output artifacts/lesson11-public-pilot `
  --registry experiments/lesson10-public-sources.json `
  --split-seed lesson11-public-v1 `
  --fineweb-bytes 6291456 `
  --openwebmath-bytes 2097152 `
  --max-documents-per-source 4096
```

生成静态预算：

```powershell
python -m nanogpt_nspire.model_budget `
  --output artifacts/lesson11-model-budget.json
```

训练 Base pilot：

```powershell
$sourceCommit = git rev-parse HEAD

python -m nanogpt_nspire.base_train `
  --data-dir artifacts/lesson11-public-pilot `
  --output-dir artifacts/lesson11-base-pilot `
  --source-commit $sourceCommit `
  --device cuda `
  --steps 1000 `
  --micro-batch-size 4 `
  --gradient-accumulation-steps 4 `
  --learning-rate 0.0006 `
  --min-learning-rate 0.00006 `
  --warmup-steps 50 `
  --eval-interval 100 `
  --eval-batches 20 `
  --overfit-gate-steps 20
```

raw documents、binary shards、logs 和 checkpoint 都在 ignored
`artifacts/`。Git 只提交代码、课程和有边界的小型 JSON evidence。

## 15. 本课能宣称与不能宣称的事

现在有证据支持：

- RTX 5080 上的 CUDA 12.8 PyTorch 环境可用；
- `6×384/context 256/vocab 264` 的真实 causal training 可运行；
- public corpus 的 source commit、row、license 和 hashes 可追溯；
- 两次 corpus build byte-identical；
- repeated-batch overfit gate 通过；
- full validation/test loss 明显优于 uniform 和 unigram baseline；
- 同 seed 两次训练的 model tensors bit-identical；
- Student 的 W4/file/RAM 静态估算通过当前门。

仍不能宣称：

- checkpoint 已会对话；
- checkpoint 能正确计算 `12×7`；
- checkpoint 能解释基础物理；
- Teacher、蒸馏或 RLVR 已提高能力；
- `.ngm v2` 实际小于 6 MiB；
- C logits 已与新 byte model 对齐；
- Nspire 已运行这个英语 checkpoint；
- Nspire 的真实速度和峰值 RAM 已通过。

## 16. 下一课

Lesson 12 将把“会续写一点英文”继续推进为“知道模型应该回答什么”：

1. 扩大并平衡高质量英语 Base/continued-pretraining 语料；
2. 加入 DeepMind Mathematics 和项目生成的可验证算术；
3. 加入短、基础的数学物理解释文本；
4. 用真实 `<USER>` / `<ASSISTANT>` token 做 supervised fine-tuning；
5. 固定算术、数学概念、物理概念和普通对话 evaluation；
6. 继续区分纯神经答案与未来 C calculator tool。

只有 Base 和 SFT 门通过后，才进入 external sequence teacher、local logit
distillation，以及同输出 token 上限的 direct/CoT RLVR 对比。
