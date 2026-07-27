# 三种可部署小模型的公平比较设计

日期：2026-07-27  
状态：已由用户确认

## 1. 为什么先补 Direct-Small

Lesson 03 的单头注意力模型和 Lesson 04 的固定 batch 过拟合模型都是教学工具，
不是正式部署候选：

- 它们没有完整的 multi-head attention、MLP 和 pre-norm block；
- Lesson 04 checkpoint 还被故意训练到严重过拟合；
- 它们没有按 4–6 MiB 部署档使用参数预算；
- 后续若直接量化或蒸馏，就没有公平的原生小模型基线。

因此在量化和蒸馏前插入 Direct-Small。它从随机参数训练，是后续
Distilled-Small 必须原样复用的 student 架构，也是所有部署指标表的第一行。

课程顺序调整为：

1. 字符、token、词表和训练样本；
2. embedding、logits 与交叉熵；
3. causal self-attention 和张量形状；
4. 完整训练循环与过拟合；
5. Direct-Small 完整小 GPT；
6. Quantized-Small；
7. Distilled-Small；
8. C 推理与 PyTorch 对齐；
9. CX II 内存和性能测量。

蒸馏后量化是单独的组合实验，不混入前三条基础路线。

## 2. 三条基础路线

| 路线 | 小模型如何得到 | 主要问题 |
|---|---|---|
| Direct-Small | 部署架构从随机参数直接训练 | 少量高精度参数能做到什么程度？ |
| Quantized-Small | 较大模型训练后转换为 INT8/INT4 | 同样存储空间下，更多低精度参数是否更强？ |
| Distilled-Small | teacher soft logits 教同一 student | 同一小架构能否因蒸馏获得更好性能？ |

“Small”在这里表示最终产物能独立满足文件和内存预算，而不是要求三条路线的
参数量相同。

## 3. 第一层公平性：同架构

Direct-Small 与 Distilled-Small 必须完全相同：

- `vocab_size=65`；
- `block_size=128`；
- `n_layer=4`；
- `n_head=5`；
- `n_embd=160`；
- MLP expansion `4×`；
- pre-norm residual block；
- bias-free Linear 和 LayerNorm；
- tanh-approximate GELU；
- tied token embedding / vocabulary head；
- FP32 部署权重；
- dropout、初始化、优化器、学习率计划和训练 token 数；
- 初始参数 seed、训练 batch 顺序和 validation windows。

两者唯一的主要实验变量是目标函数：

```text
Direct:
    hard-label cross-entropy

Distilled:
    weighted hard-label CE
    + weighted teacher/student soft-logit loss
```

这层比较回答蒸馏本身是否改善了同一个 student。若蒸馏需要修改架构、训练 token
预算或额外数据，必须成为另一项实验，不能覆盖基础对照。

## 4. 第二层公平性：同部署预算

三条路线共同遵守：

```text
目标权重档：4–6 MiB
权重文件硬上限：6,291,456 bytes
推理峰值 RAM 硬上限：25,165,824 bytes
context 上限：128 tokens
```

Direct-Small 和 Distilled-Small 使用 FP32。Quantized-Small 可以有更多参数，
但必须同时满足：

- 模型文件包含真正的 INT8/INT4 tensor；
- C runtime 的主要权重计算直接消费整数表示；
- 启动后不能把全部权重恢复为 FP32 常驻内存；
- scale、zero point、padding 和对齐开销全部计入文件；
- 反量化 scratch、KV cache 和 allocator 开销全部计入峰值 RAM。

“PyTorch checkpoint 变小”不是通过量化门槛的证据。最终以统一部署文件和
Host/Nspire runtime 实测为准。

## 5. Direct-Small v1 架构

候选比较：

| 候选 | 参数量 | FP32 原始权重 | 取舍 |
|---|---:|---:|---|
| `4×128, 4 heads` | 812,288 | 3.099 MiB | 未充分使用目标文件档 |
| `4×160, 5 heads` | 1,261,120 | 4.811 MiB | 采用；head dim 32，保留文件余量 |
| `6×144, 4 heads` | 1,522,656 | 5.808 MiB | 头部/对齐余量太小，block 更串行 |

采用 `4×160, 5 heads`。一个 block 包含：

```text
x = x + MultiHeadCausalAttention(LayerNorm(x))
x = x + MLP_GELU(LayerNorm(x))
```

完整数据流：

```text
token IDs
  -> tied token embedding + learned position embedding
  -> dropout
  -> 4 pre-norm Transformer blocks
  -> final LayerNorm
  -> tied vocabulary projection
  -> logits
```

参数公式（bias-free、tied embedding）：

```text
token embedding                         = V*C
position embedding                      = T*C
每个 block 的 QKV/O 与 MLP             = 12*C²
每个 block 的两个 LayerNorm weight      = 2*C
final LayerNorm weight                  = C

P = V*C + T*C + L*(12*C² + 2*C) + C
  = 65*160 + 128*160 + 4*(12*160² + 2*160) + 160
  = 1,261,120
```

FP32 原始权重：

```text
1,261,120 * 4 = 5,044,480 bytes = 4.811 MiB
```

预留 64 KiB 给统一文件头、词表、tensor table 和对齐后，估算为
`5,110,016 bytes`，仍低于 6 MiB 硬上限。这个估算不是最终导出文件实测。

## 6. 初步内存预算

FP32 KV cache 静态大小：

```text
K + V = 2 * L * T * C * 4 bytes
      = 2 * 4 * 128 * 160 * 4
      = 655,360 bytes
```

权重加 KV cache 约 `5.44 MiB`，距离 24 MiB 上限仍有空间。但现在还没有 C
allocator、scratch arena、程序段和 Ndless 运行时实测，因此只能标记为：

```text
static budget estimate: eligible
Host measured peak RAM: pending
Nspire measured peak RAM: pending
```

只有 Lesson 08/09 的实际加载与生成测量能关闭内存验收门。

## 7. 固定训练与评估协议

Direct-Small v1 和基础 Distilled-Small 固定：

```text
dataset: Tiny Shakespeare character-level
dataset hashes: Lesson 01 manifest
seed: 1337
training batch seed: 1339
validation seed: 1338
sample seed: 1340
batch size: 64
block size: 128
optimizer steps: 5000
training tokens: 40,960,000
optimizer: AdamW
betas: (0.9, 0.99)
weight decay: 0.1, only matrix/embedding weights
max learning rate: 1e-3
warmup: 100 steps
cosine minimum learning rate: 1e-4
gradient norm cap: 1.0
dropout: 0.1
validation batches: 50 fixed windows
sample prompt: token 0, the newline character
sample length: 300 new characters
temperature: 0.8
```

定期评估选择 validation loss 最低的 checkpoint。最终实验同时记录最后一步模型
和被选 checkpoint，避免把“训练到最后”等同于“验证最好”。

Quantized-Small 的较大 FP32 source model使用相同数据、context、validation
windows 和采样协议，但它不属于“同架构”比较，可以使用单独记录的训练 token
预算。基础 teacher 候选暂定 `81,920,000` token；Direct/Distilled student
仍固定 `40,960,000` token。最终表必须显式展示这个训练成本差异。若需要研究
同训练预算，则作为补充对照，不能覆盖同部署预算主比较。

Direct-Small 实测 selected validation loss 为 `1.4997899746894836` 后、teacher
训练开始前，teacher 质量门冻结为至少低 `0.02`：

```text
teacher selected validation loss <= 1.4797899746894836
```

未通过时，它既不能作为基础蒸馏 teacher，也不能因“参数更多”就自动升级为
Quantized-Small 候选。

## 8. 固定评价表

每条路线最终记录：

| 类别 | 字段 |
|---|---|
| 质量 | validation loss、BPC、固定 seed 样例 |
| 架构 | 层数、宽度、头数、context、参数量 |
| 存储 | checkpoint、原始权重、统一部署文件 bytes |
| RAM | 静态预算、Host peak、Nspire tracked heap |
| 训练 | token 数、优化器、训练时间、最佳 step |
| 导出 | 导出时间、量化元数据和完整性校验 |
| 对齐 | logits 最大绝对/相对误差、greedy token 一致性 |
| 速度 | Host 首 token/字符每秒、Nspire 首 token/字符每秒 |

尚未到达的阶段使用带原因的 `pending`，不能写 `0`、估计值或模拟器数字冒充真机
结果。生成文本只作定性辅助。

## 9. 验收边界

Direct-Small 基线至少需要：

- 参数公式与实际唯一参数量一致；
- tied weights 共享同一参数；
- causal future-isolation 测试通过；
- 完整 block 能在 toy 数据上反向传播和过拟合；
- 训练 token、seed、数据哈希、代码提交齐全；
- 最佳 checkpoint 严格回载后 validation loss 和固定样例可复现；
- 估算部署文件不超过 6 MiB。

它在 C exporter 和 runtime 完成前只能称为“PyTorch 侧预算合格候选”，不能称为
“已在 Nspire 部署”。
