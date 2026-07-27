# Lesson 06：Teacher、INT4 量化与一次有价值的失败

本课完成了两件彼此独立的事：

1. 训练一个 `6 layers × 6 heads × width 384` 的 FP32 Teacher；
2. 把它的二维权重压成 groupwise signed INT4，并测量体积和质量损失。

结果不能简单写成“Quantized-Small 成功”：

```text
Teacher 比 Direct-Small 更好                 是
Teacher 达到预注册质量门                    否
INT4 自身的 loss 退化门                     是
INT4 自身的 6 MiB 文件预算门                是
可晋升为正式 Quantized-Small                否
```

所以最终 artifact 的路线名是 `Quantized-Small-Diagnostic`。这不是文字游戏：
Teacher 选择失败和量化失败是两个不同问题，实验必须把它们分开。

## 1. Teacher 到底是什么

nanoGPT 是代码仓库，不是一个固定“33 MB 模型”。本项目的 Teacher 是用同一套
GPT 实现构造的电脑端训练模型：

```text
vocabulary        65 characters
context           128
layers            6
heads             6
embedding width   384
MLP ratio         4
dropout           0.2
bias              false
tied embedding    true
parameters        10,695,936
raw FP32 bytes    42,783,744
```

它本身不会直接塞进 Nspire。它有两个潜在用途：

- 量化后成为同文件预算下参数更多的模型；
- 在下一课输出 soft probability，指导 Direct-Small 架构的 student。

Teacher 必须先证明比 Direct-Small 足够强，否则它没有资格担任这两个角色。

## 2. 为什么先冻结质量门

Direct-Small 的 validation loss 是：

```text
1.4997899746894836
```

我们在 Teacher 训练前要求至少改善 `0.02`：

```text
Teacher loss <= 1.4797899746894836
```

如果看到结果后把门槛改成 `0.015`，就等于让数据替我们制定规则。那样得到的是
“这次刚好能过”的故事，不是可审计实验。

Teacher 使用：

```text
steps            10,000
batch size       64
context          128
training tokens  81,920,000
validation seed  1338
sample seed      1340
```

这是 Direct-Small 两倍的训练 token，因此以后做部署预算比较时必须把额外训练
成本一起报告。

## 3. Teacher 的真实结果

| 指标 | Direct-Small | Teacher v1 |
|---|---:|---:|
| 参数量 | 1,261,120 | 10,695,936 |
| 训练 token | 40,960,000 | 81,920,000 |
| selected validation loss | 1.499790 | 1.483962 |
| selected validation BPC | 2.163740 | 2.140904 |

Teacher 改善了 `0.015828`，相对约 `1.055%`，但距离预注册门还差
`0.004172`。因此：

```text
quality_gate_passed = false
```

最佳 checkpoint 出现在 step 2250；到 step 10000，validation loss 已回升到
`1.774067`。training loss 仍很低，说明更宽模型很快记住训练集，泛化却变差。
这是典型的过拟合，而不是“模型越大就一定越好”。

## 4. bit、整数范围与 nibble

一个 bit 只有两种状态。四个 bit 能表达：

```text
2^4 = 16
```

种编码。用二进制补码解释时，signed INT4 的完整编码范围是：

```text
[-8, 7]
```

四个 bit 也称一个 nibble。一个 byte 有八个 bit，所以能放两个 INT4：

```text
byte = high nibble | low nibble
```

本项目冻结为 low-nibble-first。例如：

```text
values [-8, -1, 7]
nibbles [0x8, 0xF, 0x7]
bytes   [0xF8, 0x07]
```

最后一个值落在第二个 byte 的低四位，高四位补零。解包时必须知道真实 value
count，不能把 padding 当成权重。

## 5. 为什么量化范围使用 `[-7,7]`

编码能够表示 `-8`，但对称量化选择：

```text
[-7, 7]
```

这样正负两侧拥有相同幅度。对一个 group：

```text
scale = max(abs(weight)) / 7
q     = clamp(round(weight / scale), -7, 7)
weight_hat = q * scale
```

如果整个 group 都是零，则：

```text
scale = 1
q = 0
```

避免除以零，同时能精确还原全零组。

量化误差来自 rounding。只要值没有额外 clipping，单个值的绝对误差通常不超过：

```text
scale / 2
```

## 6. 为什么不是每个 tensor 只用一个 scale

若整个大矩阵共享一个 scale，少数离群大权重会放大步长，使多数小权重被粗糙地
舍入。每个权重单独一个 scale 又会让 metadata 大到失去压缩意义。

本课沿最后一维每 64 个值分组：

```text
group_size = 64
```

每组共享一个 FP32 scale。这是在误差和 metadata 之间的折中。最后不足 64 的
group 会补零，但原始 shape 单独保存，反量化后会裁掉 padding。

这里的 Teacher 矩阵最后一维都恰好可被 64 整除，因此本次没有实际 padding
开销；实现和测试仍覆盖 partial final group，不能依赖这次形状的巧合。

## 7. 哪些参数量化

冻结策略：

```text
所有唯一二维参数     groupwise signed INT4 + FP32 scales
所有唯一一维参数     FP32 passthrough
```

二维参数包括 embedding、attention projection 和 MLP matrix。一维参数主要是
LayerNorm scale。

LayerNorm 向量总共只有 `4,992` 个值，即 `19,968 bytes`。保留 FP32 几乎不影响
总文件，却避免非常窄的归一化参数被 4-bit 误差放大。

## 8. tied weight 只能保存一次

Teacher 的：

```text
token_embedding.weight
lm_head.weight
```

是同一个 Parameter。量化包只保存前者，另记：

```json
{
  "lm_head.weight": "token_embedding.weight"
}
```

如果把两个 state-dict 名称各保存一次，会浪费空间，也可能在反量化后产生两个
略有不同的矩阵。独立回载确认二者仍共享同一物理 Parameter。

## 9. 实际存储账本

| 内容 | bytes |
|---|---:|
| packed INT4 matrix nibbles | 5,345,472 |
| 167,046 个 FP32 scales | 668,184 |
| FP32 LayerNorm vectors | 19,968 |
| logical payload | 6,033,624 |
| 加 64 KiB metadata reserve | 6,099,160 |
| PyTorch artifact 实际大小 | 6,057,219 |
| 6 MiB 上限 | 6,291,456 |

实际 artifact 剩余 `234,237 bytes`；按更保守的 logical payload 加固定 reserve
计算，还剩 `192,296 bytes`。

Lesson 05 的 `5,347,968 bytes` 是把所有参数都按半 byte 计算的理想下界。真实
格式还需要 scales，并特意让 LayerNorm 保持 FP32，所以不能拿理想下界冒充文件
大小。

## 10. 量化误差的三种观察层

### 权重误差

```text
weight max absolute error = 0.0361413
weight RMSE               = 0.00439817
```

这是最靠近量化公式的误差。

### logits 误差

固定 probe 上：

```text
logits max absolute error = 1.477705
logits RMSE               = 0.234576
```

最大误差看起来不小，因为几十层矩阵运算会累积局部误差，而且单个极端 logit
不能代表整体预测分布。最终仍要看交叉熵。

### validation loss 误差

| 模型 | validation loss | BPC |
|---|---:|---:|
| FP32 Teacher | 1.483962 | 2.140904 |
| dequantized INT4 reference | 1.488851 | 2.147958 |

绝对退化：

```text
0.0048893
```

相对退化：

```text
0.3295%
```

小于预注册上限 `0.05`，所以 INT4 质量门通过。量化后的 diagnostic 模型仍比
Direct-Small loss 低 `0.010939`，但这不能推翻 Teacher 先验门失败的事实。

## 11. 为什么生成文本不同

Teacher 与量化参考都使用 seed 1340 和 temperature 0.8，但 sampling 是逐 token
反馈的。一处很小的 probability 变化就可能抽到不同字符，之后上下文也不同，
整段文本迅速分叉。

因此：

- 固定 seed 文本适合检查同一 artifact 是否可复现；
- 不适合要求 FP32 与 INT4 整段文本逐字相同；
- 两模型质量比较主要看固定 validation windows 的 loss/BPC。

## 12. packed checkpoint 不等于整数推理

当前 `.pt` 确实没有原始 FP32 matrix，二维权重以两个 nibble 每 byte 保存。但
PyTorch 参考路径在计算前执行：

```text
packed INT4
  -> unpack signed integers
  -> multiply FP32 scales
  -> reconstruct FP32 matrices
  -> ordinary PyTorch FP32 inference
```

它只隔离了 weight quantization error，没有测量：

- activation quantization；
- W4A8 matrix kernel 的 rounding/saturation；
- int32 accumulator；
- requantization；
- C 端近似函数；
- packed weights 是否能边读边算而不展开整模型；
- Host 或 Nspire 峰值 RAM 与速度。

所以 runtime 状态仍是：

```text
integer C runtime   pending Lesson 08
Nspire measurement pending Lesson 09
```

只有 C kernel 直接消费 packed INT4，并证明不会先展开成完整 FP32 权重，才能满足
三路线比较对 Quantized-Small 的定义。

## 13. 为什么保留 diagnostic artifact

若因 Teacher 失败就完全不做量化，我们只能知道“整个路线没过”，却不知道问题
发生在哪一层。显式 diagnostic 模式让我们得到：

```text
Teacher gate        failed
INT4 quality gate   passed
INT4 size gate      passed
overall candidate   failed
```

CLI 默认仍拒绝失败的 Teacher。只有显式传入：

```text
--diagnostic-allow-failed-teacher
```

才会生成 `Quantized-Small-Diagnostic`，而且 `candidate_gate_passed` 永远包含
Teacher gate，不可能因量化门通过而被误标为正式候选。

## 14. 独立复核

Teacher：

- checkpoint 严格回载，无 missing/unexpected key；
- 参数量重新计算为 `10,695,936`；
- embedding/head 对象身份保持共享；
- CUDA 固定窗口 loss 精确重现；
- 301 字符固定 seed 样例精确重现；
- future-token isolation 最大误差为 `0.0`；
- checkpoint SHA-256 与记录一致。

INT4：

- 用 `weights_only=True` 从磁盘安全回载；
- 39 个 canonical tensor 和 1 个 tied alias；
- 包中不存在原始 FP32 matrix copy；
- 反量化模型严格回载且保持 weight tying；
- validation loss 与固定 seed 样例精确重现；
- future-token isolation 最大误差为 `0.0`；
- artifact SHA-256 与记录一致。

完整证据位于：

- [`lesson06-teacher.json`](../../experiments/lesson06-teacher.json)
- [`lesson06-int4.json`](../../experiments/lesson06-int4.json)

## 15. 下一步

Teacher v1 不能进入正式蒸馏。下一课应先预注册一个新的 Teacher v2 协议，针对
本次“step 2250 后快速过拟合”的证据改善正则化或训练选择；v1 的失败结果永久
保留，不能覆盖。

只有新 Teacher 通过独立质量门，才同时解锁：

- 正式 Quantized-Small；
- 与 Direct-Small 同架构、同 student 训练预算的 Distilled-Small。

蒸馏后再量化仍作为第四个组合增强实验，不能混进三条基础路线。
