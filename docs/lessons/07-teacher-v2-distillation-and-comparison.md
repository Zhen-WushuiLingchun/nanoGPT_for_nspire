# Lesson 07：Teacher v2、正式量化与蒸馏

本课不是简单地把三个脚本都跑一遍，而是连续回答四个问题：

1. Teacher v1 为什么在 step 2250 后快速过拟合？
2. 只把 dropout 从 `0.2` 提高到 `0.3`，Teacher v2 会怎样变化？
3. v1 和 v2 架构相同、量化算法也相同，为什么 INT4 误差仍然不同？
4. 通过质量门的 Teacher v2 能否把同一个 Direct-Small student 教得更好？

最后一个问题得到了一个很有价值的负结果：

```text
基础 5000-step 蒸馏没有胜过 Direct-Small
延长到 10000 steps 后明显改善，但仍未胜过 Direct-Small
```

所以我们既没有把蒸馏写成“必然有效”，也没有因第一次失败就断言“蒸馏无效”。
本课把固定预算结果和延长训练的扩展实验分开记录。

## 1. 先冻结 Teacher v2，而不是边跑边调

Teacher v1 的预注册质量门是：

```text
selected validation loss <= 1.4797899746894836
```

它的实际结果是：

```text
best step                2250
selected validation loss 1.4839615273475646
final validation loss    1.7740670037269592
quality gate             failed
```

Teacher v1 确实优于 Direct-Small 的 `1.4997899746894836`，但改善没有达到事先
要求的 `0.02`，而且后半程发生严重过拟合。

根据这个现象，Teacher v2 只修改一个训练变量：

```text
dropout: 0.2 -> 0.3
```

其余全部保持：

```text
layers            6
heads             6
embedding width   384
context           128
parameters        10,695,936
seed              1337
steps             10,000
batch size        64
training tokens   81,920,000
optimizer         AdamW
learning rate     unchanged
batch order       unchanged
validation windows unchanged
quality gate      unchanged
```

输出目录和路线名当然需要不同，否则会覆盖 v1；它们不参与数值计算。

这里故意没有同时修改 weight decay、模型宽度、训练步数或 early stopping。
一次改多个变量也许能得到更好数字，却无法知道收益来自哪里。

## 2. dropout 在 Transformer 中做了什么

dropout 在训练时随机把一部分激活置零，并对保留部分重新缩放，使期望值大致不变。
本项目的 dropout 出现在：

- token 与 position embedding 相加之后；
- attention probability；
- attention output；
- MLP output。

从 `0.2` 提高到 `0.3`，意味着每次训练前向都会屏蔽更多临时连接。模型不能总依赖
同一组特征共同出现，必须学出更冗余、更稳健的表示。

这通常会产生一对相反效果：

```text
训练前期：有效容量更小、噪声更强、学习更慢
训练后期：不容易快速记住训练样本、泛化可能更好
```

在 `model.eval()` 的验证和推理阶段，dropout 会关闭。因此 v2 的改善不是“验证时
随机丢掉 30% 神经元”，而是训练期间的随机屏蔽改变了最终学到的参数。

## 3. v1 与 v2 的真实曲线

| step | Teacher v1 loss | Teacher v2 loss | 当时更好 |
|---:|---:|---:|---|
| 0 | 4.142343 | 4.142343 | 完全相同 |
| 2250 | **1.483962** | 1.508913 | v1 |
| 2500 | **1.489955** | 1.499097 | v1 |
| 2750 | **1.487051** | 1.488038 | v1 |
| 3000 | **1.487447** | 1.488569 | v1 |
| 3250 | 1.486702 | **1.479078** | v2 |
| 3500 | 1.487489 | **1.474331** | v2 |
| 4000 | 1.497527 | **1.473149** | v2 |
| 4500 | 1.509579 | **1.463887** | v2 |
| 10000 | 1.774067 | **1.546478** | v2 |

step 0 的 validation loss 精确相同，初始参数也逐 tensor 相同。这是很重要的
实验控制：它排除了“v2 只是随机初始化更幸运”。

曲线正好展示了 dropout 的两面性：

1. v2 在 step 2250 仍明显更差，说明更强正则化拖慢了前期拟合；
2. 到 step 3250，v2 开始超过 v1；
3. v1 的最佳点是 step 2250，v2 延后到 step 4500；
4. v2 最佳 loss 降到 `1.4638866233825683`；
5. v2 到 step 10000 也出现回升，所以 dropout 只是延缓、减轻过拟合，没有消灭
   过拟合。

Teacher v2 相比 v1 改善：

```text
1.4839615273475646 - 1.4638866233825683
= 0.0200749039649963
```

它超过质量门的余量为：

```text
1.4797899746894836 - 1.4638866233825683
= 0.0159033513069153
```

因此：

```text
Teacher v2 quality_gate_passed = true
```

这同时解锁正式 INT4 候选和蒸馏实验。

## 4. 为什么 Teacher v2 仍然会过拟合

Tiny Shakespeare 的训练 split 只有约一百万字符，而 Teacher 有约 1070 万参数。
参数量不是训练样本信息量的直接等价物，但这个比例足以提示：Teacher 很容易拥有
记忆具体训练片段的容量。

dropout `0.3` 增加了学习难度，却没有增加新数据。训练足够久后，模型仍能找到适应
dropout 噪声的记忆方式。因此 v2 的结论应写成：

```text
更强 dropout 延后并减轻了过拟合
```

而不能写成：

```text
更强 dropout 解决了过拟合
```

本实验还不能说明 `0.3` 是全局最优 dropout。它只证明，在冻结的 v1/v2 对比中，
`0.3` 比 `0.2` 更适合当前数据、架构和训练预算。

## 5. Teacher v2 的正式 INT4 候选

量化规则与 Lesson 06 完全相同：

```text
二维矩阵       groupwise signed INT4
量化范围       [-7, 7]
group size     64
scale          每组一个 FP32
一维参数       保持 FP32
tied embedding 只保存一份
```

Teacher v2 的结果：

| 指标 | FP32 v2 | INT4 v2 |
|---|---:|---:|
| validation loss | 1.463887 | 1.473799 |
| BPC | 2.111942 | 2.126243 |

量化造成：

```text
absolute loss degradation 0.00991249561309826
relative degradation      0.6771354731142833%
```

仍低于预注册上限 `0.05`。存储账本为：

| 内容 | bytes |
|---|---:|
| packed INT4 nibbles | 5,345,472 |
| FP32 scales | 668,184 |
| FP32 一维参数 | 19,968 |
| logical payload | 6,033,624 |
| 加 64 KiB reserve | 6,099,160 |
| 实际 PyTorch artifact | 6,057,283 |
| 6 MiB 上限 | 6,291,456 |

Teacher 门、量化误差门和体积门全部通过，所以产物路线可正式标为：

```text
Quantized-Small
```

但它仍不是完整的 Nspire 整数推理结论。当前 PyTorch 参考会把 packed INT4
反量化成 FP32 矩阵再计算；Lesson 08 需要让 C kernel 直接消费整数权重，并测量
真实峰值 RAM。

## 6. 为什么 v1 与 v2 的量化误差不同

这是本课很容易混淆的地方。v1 和 v2 的模型形状相同，量化算法也相同，所以它们的
logical payload 完全相同；但“相同 shape”不等于“相同 weight values”。

实测对照如下：

| 指标 | INT4 v1 diagnostic | INT4 v2 formal |
|---|---:|---:|
| FP32 source loss | 1.483962 | **1.463887** |
| INT4 loss | 1.488851 | **1.473799** |
| loss degradation | **0.004889** | 0.009912 |
| weight max absolute error | **0.036141** | 0.055036 |
| weight RMSE | **0.004398** | 0.005349 |
| logits max absolute error | **1.477705** | 1.668247 |
| logits RMSE | **0.234576** | 0.308941 |

看起来有点反直觉：

```text
v2 的 FP32 模型更好，但 v2 的量化退化反而更大。
```

原因应分成三个层次理解。

### 6.1 dropout 不在推理时直接参与，但会改变权重

v2 的 dropout 在推理时已经关闭，量化器也没有量化 dropout。可是训练时更强的
随机屏蔽改变了优化路径；再加上 v1 选择 step 2250、v2 选择 step 4500，两个
checkpoint 的实际权重分布并不相同。

### 6.2 groupwise scale 对组内最大绝对值敏感

每组的量化步长是：

```text
scale = max(abs(weight_in_group)) / 7
```

若一个组出现更大的绝对值，整个组的 scale 会变大，其余较小权重的 rounding
网格也随之变粗。即使参数量、group size 和文件大小不变，误差仍可能增大。

本次 v2 的 weight max error 和 RMSE 都高于 v1，说明差异已经存在于权重量化层，
随后经过多层 attention 与 MLP 传播，形成更大的 logits RMSE。

### 6.3 更好的 FP32 模型不保证更“耐量化”

FP32 validation loss 衡量预测质量；量化退化衡量模型对参数扰动的敏感度。二者是
不同性质。一个 FP32 checkpoint 可以预测更准，却位于对低比特 rounding 更敏感的
参数区域。

我们现在有证据说：

- v2 的权重级、logits 级和 loss 级量化误差都更大；
- 相同算法和 shape 排除了格式差异；
- v2 仍因更好的 FP32 起点而得到更好的最终 INT4 loss。

我们还没有逐组统计证明“具体哪几个离群权重组”主导了误差，所以不能把 scale
机制写成已经定位完成的唯一根因。那需要额外做 per-group range、clipping 与
sensitivity ablation。

虽然 v2 多损失了约 `0.00502` 的量化精度优势，它的 FP32 起点比 v1 好约
`0.02007`，最终 INT4 v2 仍比 INT4 v1 好约 `0.01505`。

## 7. 蒸馏到底在传递什么

普通训练只告诉 student 正确 token 的编号。例如正确字符是 `e`，hard label
只表达：

```text
e 是目标，其余字符不是
```

Teacher 的完整 logits 还表达错误选项之间的相对关系。例如在某个上下文中：

```text
P(e) 很高
P(a) 和 P(i) 次之
P(换行) 很低
```

这些非目标类别之间的结构常被称为 dark knowledge。蒸馏希望 student 不只记住
答案，还学习 Teacher 怎样分配不确定性。

本课冻结：

```text
temperature T = 2.0
soft weight α = 0.5
hard weight   = 1 - α = 0.5
```

损失为：

```text
hard = CrossEntropy(student_logits, targets)

soft = T² * KL(
    softmax(teacher_logits / T)
    ||
    softmax(student_logits / T)
)

total = (1 - α) * hard + α * soft
```

### 为什么除以 temperature

`T > 1` 会把概率分布变平，让原本很小的非目标概率更容易被 student 看见。

### 为什么再乘 `T²`

温度缩放会让 softmax 和 log-softmax 的梯度变小。经典蒸馏乘 `T²`，用于补偿
梯度尺度变化。

### 为什么 KL 方向是 Teacher 到 student

PyTorch 实现使用 Teacher probability 作为 target、student log probability
作为 input，计算：

```text
KL(teacher || student)
```

它惩罚 student 没有覆盖 Teacher 认为可能的字符。

### 为什么先 flatten `(B,T,V)`

每个字符位置都是一个分类样本。实现先把：

```text
(batch, time, vocabulary)
```

展平成：

```text
(batch * time, vocabulary)
```

再使用 `batchmean`。否则如果只把 batch 当分母，soft loss 会额外乘上 sequence
length，`α=0.5` 就不再具有预期含义。

Teacher 始终处于 `eval()` 和 inference mode，所有参数 `requires_grad=False`；
反向传播只更新 student。最终 student checkpoint 也不包含 Teacher 权重。

## 8. 公平的基础蒸馏结果

Direct-Small 与 Distilled-Small 使用完全相同的：

```text
4 layers
5 heads
width 160
dropout 0.1
1,261,120 parameters
FP32
seed 1337
5000 steps
40,960,000 student training tokens
optimizer and batch order
```

两者 step 0 loss 都是：

```text
4.209567728042603
```

初始参数逐 tensor 相同。唯一主要变量是训练目标。

| step | Direct-Small loss | Distilled-Small loss |
|---:|---:|---:|
| 250 | **2.282485** | 2.346696 |
| 1000 | **1.758442** | 1.790053 |
| 2500 | **1.558792** | 1.588184 |
| 4000 | **1.510460** | 1.537329 |
| 5000 | **1.499790** | 1.522163 |

最终：

```text
Direct-Small loss     1.4997899746894836
Distilled-Small loss  1.5221625304222106
distillation gap      -0.022372555732727006
relative gap          -1.4917125804470734%
```

这里的负号表示“改善”为负，即蒸馏更差。

基础蒸馏的 best step 恰好是最后的 step 5000，validation loss 全程总体下降，并未
出现 Direct/Teacher 那种明显后期回升。因此这次失败更像收敛速度不足，而不是
student 已经过拟合。

为什么可能更慢？

- hard-label 梯度权重从 `1.0` 降到 `0.5`；
- student 同时要满足真实 token 和 Teacher 分布两个目标；
- Teacher 只比 Direct-Small 好 `0.0359` loss，soft signal 的净收益可能不够大；
- 当前 `T=2, α=0.5` 未必适合这个极小字符模型。

这些是由结果支持的候选解释，不是已被单次实验逐一证明的根因。

另外，distillation training loss 是 hard 与 soft 的加权和，不能直接和 Direct
的纯 cross-entropy training loss 比大小。公平比较仍使用相同固定 validation
windows 上的 hard cross-entropy。

## 9. 10000-step 扩展实验

用户允许延长训练，但它必须只作为扩展对比。为了不丢失 AdamW 动量，我们没有从
step 5000 的纯模型 checkpoint 重新创建 optimizer，而是：

1. 从同一个 seed 重新运行；
2. 前 5000 步使用与基础蒸馏完全相同的学习率曲线；
3. step 5000 后保持最低学习率 `0.0001`；
4. 连续训练到 step 10000。

扩展 run 在 step 5000 精确复现：

```text
base run step 5000      1.5221625304222106
extended run step 5000  1.5221625304222106
```

这证明前半程轨迹与基础实验一致。

| step | extended validation loss |
|---:|---:|
| 5000 | 1.522163 |
| 7000 | 1.516432 |
| 9000 | 1.513637 |
| 9500 | 1.508344 |
| 9750 | **1.506599** |
| 10000 | 1.512509 |

延长训练把基础蒸馏改善了：

```text
1.5221625304222106 - 1.5065994238853455
= 0.015563106536865101
```

因此“基础版还没有收敛完”得到支持。但扩展版仍落后 Direct-Small：

```text
1.5065994238853455 - 1.4997899746894836
= 0.006809449195861905
```

而且它用了两倍 student training token。step 10000 的回升还显示，软目标训练最终
也开始出现过拟合。

所以扩展实验的正确结论是：

```text
延长训练显著缩小差距，但没有使当前蒸馏方案胜过 Direct-Small；
该结果不属于同 student-token 预算的基础三路线比较。
```

## 10. 三种小模型的当前公平比较

| 路线 | 参数量 | 权重精度 | source/student 训练 token | validation loss | BPC | 当前模型 artifact |
|---|---:|---|---:|---:|---:|---:|
| Direct-Small | 1,261,120 | FP32 | 40,960,000 | 1.499790 | 2.163740 | 5,096,641 B |
| Quantized-Small | 10,695,936 | INT4 matrix + FP32 scales/norm | 81,920,000 | **1.473799** | **2.126243** | 6,057,283 B |
| Distilled-Small | 1,261,120 | FP32 | 40,960,000 | 1.522163 | 2.196016 | 5,097,383 B |

解释这张表时要保留两个边界：

1. Direct 与 Distilled 是严格的同架构、同 student token 对比，当前 Direct 获胜；
2. Quantized 的存储与 PyTorch 参考质量门已通过，但真正的整数 C runtime、峰值 RAM
   和 Nspire 速度仍待 Lesson 08/09 测量。

10,000-step Distilled-Small-Extended 的 loss 为 `1.506599`，只作为扩展行，不放进
基础表冒充同预算结果。

## 11. 训练成本也必须报告

在同一台 RTX 5080 Laptop GPU 上：

| run | optimizer update seconds | training token |
|---|---:|---:|
| Direct-Small | 74.68 | 40,960,000 |
| Distilled-Small | 195.82 | 40,960,000 |
| Distilled-Small-Extended | 428.17 | 81,920,000 |
| Teacher v2 | 574.66 | 81,920,000 |

蒸馏每个 batch 额外执行 Teacher 前向，所以即使 student 参数量不变，训练也明显
更贵。部署时只带 student，不需要 Teacher，因此这个成本影响训练时间，不影响
student 模型文件大小。

这里的 peak CUDA memory 是训练测量，不是 Nspire 推理峰值 RAM：

```text
Direct-Small training peak       606,734,848 bytes
Distilled-Small training peak    770,766,848 bytes
Teacher v2 training peak       1,813,156,352 bytes
```

## 12. 独立复核

Teacher v2：

- checkpoint 严格回载；
- 参数量重算为 `10,695,936`；
- tied embedding/head 对象身份保持；
- 固定窗口 loss 与固定 seed sample 精确重现；
- future-token isolation 最大误差 `0.0`；
- checkpoint bytes 与 SHA-256 一致。

INT4 v2：

- 使用 `weights_only=True` 安全回载；
- 不含原始 FP32 matrix copy；
- 反量化参考保持 weight tying；
- 固定 loss、sample 和因果隔离精确重现；
- 体积门、量化质量门和 Teacher 门全部通过。

基础与扩展 Distilled-Small：

- student 架构严格等于 Direct-Small；
- checkpoint 中不含 Teacher 权重；
- 严格回载无 missing/unexpected key；
- 固定 loss、sample、因果隔离和文件哈希精确重现；
- 扩展 run 在 step 5000 与基础 run 精确相同。

机器可读证据：

- [`lesson07-teacher-v2.json`](../../experiments/lesson07-teacher-v2.json)
- [`lesson07-int4.json`](../../experiments/lesson07-int4.json)
- [`lesson07-distilled-small.json`](../../experiments/lesson07-distilled-small.json)
- [`lesson07-distilled-small-extended.json`](../../experiments/lesson07-distilled-small-extended.json)
- [`small-model-comparison.json`](../../experiments/small-model-comparison.json)

## 13. 下一步

基础三路线已经各自得到一个可独立部署的模型候选或基线。下一课进入 C 推理与
PyTorch 对齐：

1. 冻结统一二进制导出格式；
2. 让 Host C 读取 Direct/Distilled FP32 权重；
3. 为 Quantized-Small 实现真正直接消费 packed INT4 的整数 kernel；
4. 对齐固定输入 logits 和固定 seed 生成序列；
5. 测量 Host C 文件大小、峰值 RAM 和字符生成速度。

蒸馏后再量化仍保留为第四个组合增强实验。它不能回头改写本课三条基础路线各自
带来的收益。
