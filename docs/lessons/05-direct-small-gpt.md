# Lesson 05：Direct-Small 完整小 GPT

前四课逐层学习了 token、embedding、交叉熵、单头 attention 和训练循环。本课
第一次把这些部件组成一个完整、按设备文件预算设计的 GPT：

```text
Direct-Small v1
4 Transformer blocks
5 attention heads
embedding width 160
head width 32
context 128
1,261,120 parameters
```

它从随机参数直接训练，不依赖 teacher，也不使用量化。这是三条小模型路线中的
第一条正式基线：

```text
Direct-Small
Quantized-Small
Distilled-Small
```

后面的 Distilled-Small 必须复用本课完全相同的 student 架构，才能隔离蒸馏
目标的贡献。

## 1. 为什么 Lesson 03 不能直接当 Direct-Small

Lesson 03 有 token/position embedding、一个 attention head 和一条 residual，
但缺少：

- multi-head attention；
- attention 前的 LayerNorm；
- MLP/GELU；
- 第二条 residual；
- 多个 block 堆叠；
- final LayerNorm；
- embedding/head weight tying；
- 针对 4–6 MiB 文件档的参数设计。

它适合观察 `(B,T,T)` attention matrix，却不是我们准备移植的完整小 GPT。
Lesson 04 checkpoint 还故意只记住 32 个标签，更不能用于正式质量比较。

## 2. 完整数据流

实现位于
[`direct_small_gpt.py`](../../training/nanogpt_nspire/models/direct_small_gpt.py)。

```text
token IDs (B,T)
  -> token embedding (B,T,C)
  +  position embedding (T,C)
  -> embedding dropout
  -> Transformer block × 4
  -> final LayerNorm
  -> tied vocabulary head
  -> logits (B,T,V)
```

固定符号：

```text
B = batch size
T = active sequence length, T <= 128
V = vocabulary size = 65
C = embedding width = 160
H = attention heads = 5
D = head width = C/H = 32
```

## 3. fused QKV 与五个注意力头

Lesson 03 分别使用三个 `C -> C` Linear。本课像 nanoGPT 一样先用一次：

```text
qkv = Linear(C, 3C)(x)
```

得到：

```text
qkv: (B,T,3C)
```

然后在最后一维切成：

```text
q, k, v: (B,T,C)
```

每个 tensor 再 reshape 和 transpose：

```text
(B,T,C)
  -> (B,T,H,D)
  -> (B,H,T,D)
```

真实配置中：

```text
(B,T,160) -> (B,5,T,32)
```

每个 head 有独立的 query/key/value 子空间。五个 head 可以同时学习不同关系，
例如短邻接、换行、标点、说话人模式或更远字符依赖。这里的“可以”表示模型容量，
不是声称某个 head 已经具有可解释语义。

## 4. 多头 causal attention

每个 head 独立计算：

```text
scores = Q @ Kᵀ / sqrt(D)
```

形状：

```text
Q:       (B,H,T,D)
Kᵀ:      (B,H,D,T)
scores:  (B,H,T,T)
```

再应用下三角 mask：

```text
future score -> -infinity
weights = softmax(scores)
```

所以：

```text
weights[b,h,t,j] = 0, when j > t
```

五个 head 各自混合 Value：

```text
context = weights @ V       # (B,H,T,D)
```

之后 transpose 并拼回：

```text
(B,H,T,D) -> (B,T,H,D) -> (B,T,C)
```

最后通过 `C -> C` output projection。测试不只查看 mask buffer，还会修改未来
token 并要求较早 logits 完全不变。

## 5. pre-norm residual block

每个 block 的结构是：

```python
x = x + attention(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```

这称为 pre-norm，因为 LayerNorm 位于子层之前。两条 residual path 分别绕过
attention 和 MLP：

```text
x -------------------- + -> hidden
  \-> LN -> attention /

hidden ---------------- + -> output
      \-> LN -> MLP    /
```

residual 让梯度拥有较直接的传播路径；pre-norm 通常也比把 norm 放在残差之后
更容易稳定训练较深 decoder。

本课的 LayerNorm 只有可训练 scale，没有 bias：

```text
y = (x - mean) / sqrt(variance + epsilon) * weight
```

这与冻结的 `bias=False` 架构一致，也减少 C runtime 中需要加载的 tensor。

## 6. MLP 与 tanh GELU

attention 负责跨位置交换信息，MLP 在每个位置独立变换通道：

```text
(B,T,160)
  -> Linear(160,640)
  -> GELU
  -> Linear(640,160)
```

扩展倍率是 4。GELU 不像 ReLU 那样把所有负值直接截断，而是平滑调节。我们显式
使用 PyTorch 的：

```python
nn.GELU(approximate="tanh")
```

它比基于 `erf` 的精确公式更适合以后在 C 中复现。现在仍需通过 logits 数值对齐
测试，不能只因公式相同就预先宣称 C 一致。

## 7. final LayerNorm

四个 block 后再执行：

```text
hidden = final_layer_norm(hidden)
```

它把最终 residual stream 规范化后再映射到 vocabulary。若省略 final norm，
模型仍可能训练，但就不再是我们为蒸馏和导出冻结的同一架构。

## 8. tied embedding 与 vocabulary head

输入 embedding 是一个 `(V,C)` 矩阵。输出 head 原本也需要一个 `(V,C)` 矩阵。
本课让它们共享同一个 Parameter：

```python
lm_head.weight = token_embedding.weight
```

它不是训练后复制，也不是每次保持数值相等，而是同一块可训练存储：

```text
token_embedding.weight is lm_head.weight
```

优点：

- 少 `V*C = 10,400` 个独立参数；
- 输入和输出字符表示共享空间；
- FP32 少 41,600 bytes；
- Direct 与 Distilled student 更容易保持完全一致。

checkpoint 的 state dict 可能出现两个名字，但重新构造模型后它们仍绑定到同一
Parameter。统一部署格式以后只写一份物理 tensor。

## 9. dropout 的训练/推理差异

本课 `dropout=0.1`，应用于：

- token + position embedding；
- attention weights；
- attention output；
- MLP output。

在 `model.train()` 下，每次随机丢弃约 10% 的相应激活并缩放其余值。在
`model.eval()` 下 dropout 完全关闭。

dropout 没有可导出的权重，也不增加推理文件大小。Direct 与 Distilled 的
dropout 配置和随机种子必须相同。future-isolation 和 checkpoint 复现实验都在
eval mode 下运行。

## 10. 参数公式

bias-free、tied embedding、MLP ratio 4 时：

```text
token embedding                    = V*C
position embedding                 = T*C

每个 block:
  QKV + attention output           = 4*C²
  MLP input + output               = 8*C²
  two LayerNorm weights            = 2*C

final LayerNorm weight             = C
```

总计：

```text
P = V*C + T*C + L*(12*C² + 2*C) + C
```

代入：

```text
V=65, T=128, L=4, C=160

P = 65*160 + 128*160
    + 4*(12*160² + 2*160)
    + 160
  = 1,261,120
```

FP32 原始参数：

```text
1,261,120 * 4 = 5,044,480 bytes = 4.811 MiB
```

单元测试同时计算公式与实际唯一 Parameter 数；任何新增 bias、取消 tying 或修改
层宽都会让测试失败。

## 11. 为什么选择 4×160

设计阶段比较了：

| 架构 | 参数 | FP32 原始权重 |
|---|---:|---:|
| 4 layers × 128 | 812,288 | 3.099 MiB |
| 4 layers × 160 | 1,261,120 | 4.811 MiB |
| 6 layers × 144 | 1,522,656 | 5.808 MiB |

`4×128` 没有充分使用 4–6 MiB 档。`6×144` 太接近 6 MiB 硬上限，给文件头、
词表、scale/对齐和格式演进留下的空间很小；六个串行 block 对 ARM 延迟也更不利。

`4×160` 的 head dimension 恰好为 32，FP32 还剩约 1.19 MiB 文件空间，因此
成为 v1。这个选择以后可以被真实 Nspire 数据推翻，但不能在三模型基础比较中途
无记录地改变。

## 12. 初步部署预算

先预留 64 KiB 给统一文件格式：

```text
estimated file
= raw FP32 weights + metadata reserve
= 5,044,480 + 65,536
= 5,110,016 bytes
```

硬上限：

```text
6,291,456 bytes
```

FP32 KV cache 静态估算：

```text
2 * layers * context * width * 4
= 2 * 4 * 128 * 160 * 4
= 655,360 bytes
```

这只说明 PyTorch 侧架构具备预算候选资格。它没有计入最终 tensor table、
allocator、scratch arena、C 程序段和 Ndless 运行时，不能代替 Lesson 08/09 的
Host 与 CX II 实测。

## 13. AdamW 参数分组

训练入口位于
[`direct_small_train.py`](../../training/nanogpt_nspire/direct_small_train.py)。

AdamW 的 weight decay 只应用于二维及以上参数：

```text
embedding/matrix weights -> weight_decay = 0.1
LayerNorm/bias vectors    -> weight_decay = 0.0
```

共享的 embedding/head Parameter 只能出现在 optimizer 中一次。测试会比较
所有对象 ID，防止 weight tying 导致重复更新。

固定 betas：

```text
beta1 = 0.9
beta2 = 0.99
```

## 14. warmup 与 cosine decay

最大学习率：

```text
1e-3
```

前 100 步线性 warmup：

```text
lr(step) = max_lr * step / 100
```

之后用 cosine 衰减，在 step 5000 到达：

```text
min_lr = 1e-4
```

warmup 避免随机初始化时立刻使用最大步长；cosine decay 让后期更新逐渐细化。
schedule 的端点和非法参数都有独立测试。

## 15. 固定训练协议

Direct-Small v1：

```text
seed                    = 1337
batch size              = 64
context                 = 128
steps                   = 5000
training tokens         = 40,960,000
dropout                 = 0.1
max gradient norm       = 1.0
validation interval     = 250
validation batches      = 50
validation seed         = 1338
training batch seed     = 1339
sample seed             = 1340
sample temperature      = 0.8
```

Distilled-Small 的基础比较必须使用同一初始化 seed、batch 顺序、更新数、优化器
和 schedule。训练目标以外的变化会破坏“唯一主要变量是蒸馏”的解释。

## 16. 为什么选择最佳 validation checkpoint

训练越久，training loss 往往继续下降，但 validation loss 可能先降后升。入口
在 step 0、每 250 步和最终 step 使用同一组 seeded validation windows：

```text
if validation_loss < best_validation_loss:
    preserve CPU state dict
```

训练完成后重新加载 best state，复算 validation loss，再生成固定 seed 样例和
写 checkpoint。

实验会同时记录：

- initial validation；
- final-step validation；
- selected/best validation；
- best step。

这样不会把“最后一次更新”错误地当成“泛化最好”。

## 17. teacher 不是 nanoGPT 仓库大小

nanoGPT 是代码库，不是一个固定的 33 MB 模型。官方默认 GPT-2 是 124M 参数；
官方 Tiny Shakespeare 示例采用 `6 layers / 6 heads / width 384`。

我们的 provisional teacher 候选沿用这个形状，但统一：

```text
vocab = 65
context = 128
bias = false
tied embeddings = true
```

估算：

```text
parameters = 10,695,936
FP32 weights = 42,783,744 bytes = 40.802 MiB
ideal packed INT4 weights = 5,347,968 bytes = 5.100 MiB
```

它只在电脑上以 FP32 训练。若 validation 明显优于 Direct-Small，它可以：

1. 作为 Distilled-Small 的 soft-logit teacher；
2. 作为 Quantized-Small 的 INT4 source model。

INT8 裸权重约 10.2 MiB，无法进入 6 MiB 档。INT4 的 scale、例外 tensor 和
文件头仍必须实测；`5.100 MiB` 只是理想 packed-weight 下界。

teacher 目前状态是 provisional。若它没有稳定优于 Direct-Small，就没有资格
进入蒸馏实验。

## 18. 测试证据

Direct-Small 模型测试覆盖：

- config 与 head divisibility；
- `(B,H,T,T)` weights 形状、归一化和 causal mask；
- future-token isolation；
- 完整 logits/loss/backward；
- contextual fixed-batch 学习；
- tied Parameter 对象身份；
- 默认参数量和 FP32 bytes；
- derived mask 不进入 state dict；
- 非法 token、dtype、形状与 context。

训练入口测试覆盖：

- warmup/cosine 端点；
- AdamW decay/no-decay 分组和唯一参数；
- 配置失败；
- CPU smoke training；
- best checkpoint、样例、部署 pending 字段和严格回载。

当前全套测试为 `97 passed`。

## 19. 真实实验命令

实现提交后使用：

```powershell
python -m nanogpt_nspire.direct_small_train `
  --data-dir artifacts/data/tinyshakespeare `
  --output-dir artifacts/lesson05-direct-small `
  --device auto `
  --seed 1337 `
  --steps 5000 `
  --batch-size 64 `
  --block-size 128 `
  --n-layer 4 `
  --n-head 5 `
  --n-embd 160 `
  --dropout 0.1 `
  --learning-rate 0.001 `
  --min-learning-rate 0.0001 `
  --warmup-steps 100 `
  --weight-decay 0.1 `
  --beta1 0.9 `
  --beta2 0.99 `
  --max-grad-norm 1.0 `
  --eval-interval 250 `
  --eval-batches 50 `
  --log-interval 100 `
  --sample-tokens 300 `
  --temperature 0.8 `
  --source-commit <implementation-commit>
```

生成：

```text
artifacts/lesson05-direct-small/
├── direct_small_gpt.pt
└── run.json
```

两者均被 Git 忽略。

## 20. 真实实验结果

实现提交后，本节将记录：

- initial、final-step 和 selected validation loss/BPC；
- best step 和固定 seed 样例；
- checkpoint 大小/hash 与严格回载；
- 实际 CUDA 时间和 peak allocated memory；
- 5,110,016-byte 初步部署估算是否通过；
- Host C、统一模型文件和 Nspire 指标仍为 pending 的边界。

## 21. 下一课：Quantized-Small

Lesson 06 会先讲量化数学和误差，再决定 provisional teacher 是否通过质量门。
只有 teacher 质量与 INT4 文件预算同时成立，才会成为 Quantized-Small。
