# Lesson 04：完整训练循环与过拟合

Lesson 03 已经让字符之间通过 causal self-attention 交换信息，也运行过一次
Tiny Shakespeare 训练。但当时训练循环作为一段完整代码出现，我们还没有逐步
检查参数究竟怎样改变。

本课保持模型结构不变，专门拆开一次优化步骤：

```text
清空旧梯度
  -> forward
  -> cross-entropy loss
  -> backward
  -> 检查并裁剪梯度
  -> optimizer.step()
  -> 测量参数更新
```

然后，我们故意让模型反复学习**完全相同的一个 batch**。目标不是得到好模型，
而是让模型把这 32 个 next-token 标签记住。这个实验称为 tiny-batch overfit
test，是训练新模型时非常有用的接线检查。

## 1. 本课目标

完成本课后，应当能够解释：

1. `model(inputs, targets)` 在 forward 阶段计算了什么；
2. 为什么 loss 必须是一个可求导的标量；
3. `loss.backward()` 如何把梯度写入每个参数的 `.grad`；
4. 为什么每一步开始必须清空旧梯度；
5. 梯度范数和 gradient clipping 分别在测量、限制什么；
6. `optimizer.step()` 改变的是参数，不是 loss 本身；
7. 为什么同一 batch 的 loss 接近 0 仍然不代表模型能泛化；
8. `train()`、`eval()` 和 `inference_mode()` 的职责区别。

本课不会加入 LayerNorm、MLP、多头注意力或更多 block。这样，实验中的新变量
只有训练过程，而不是模型结构。

## 2. 固定 batch 是什么

普通训练会不断随机抽取窗口：

```text
step 1 -> batch A
step 2 -> batch B
step 3 -> batch C
```

不同 batch 难度不同，所以相邻 training loss 会自然抖动。本课只调用一次
`make_batch`：

```text
seeded make_batch -> fixed inputs, fixed targets

step 1 -> fixed batch
step 2 -> fixed batch
...
step N -> fixed batch
```

输入和目标形状仍为：

```text
inputs:  (B,T)
targets: (B,T)
```

真实实验使用 `B=1,T=32`，总共只有 32 个标签。模型知道每个 token 的位置，
所以它有足够容量记忆这一小段序列。

## 3. forward：从 token 到标量 loss

模型调用：

```python
logits, loss = model(inputs, targets)
```

返回形状：

```text
logits: (B,T,V)
loss:   scalar
```

其中 `V=65`。每个 `(b,t)` 位置都有一个长度为 65 的 logits 向量。交叉熵先对
词表维度做 log-softmax，再取正确 target 的负对数概率，最后对 `B*T` 个位置
求平均：

```text
loss = -(1 / (B*T)) * Σ log p(correct token | context)
```

loss 越小，模型分给正确字符的概率越大。自然对数交叉熵除以 `ln(2)` 就得到
bits per character：

```text
BPC = loss / ln(2)
```

## 4. computation graph 与 backward

PyTorch 在 forward 时记录：

- 哪些张量参与运算；
- 每个运算的输入输出关系；
- 哪些叶子张量是可训练参数；
- backward 时需要哪些中间结果。

如果参数为 `θ`，标量 loss 为 `L`，则：

```text
loss.backward()
```

计算：

```text
∂L/∂θ
```

并把结果累加到 `θ.grad`。这是 reverse-mode automatic differentiation：
从标量 loss 反向沿计算图应用链式法则。我们不手写整条 Transformer 的导数，
但数学上仍然是在逐层计算：

```text
∂L/∂θ = ∂L/∂logits
       × ∂logits/∂hidden
       × ...
       × ∂intermediate/∂θ
```

## 5. 为什么要先 zero_grad

PyTorch 默认对 `.grad` 做累加：

```text
第一次 backward: grad = g1
第二次 backward: grad = g1 + g2
```

梯度累加在大模型训练中有用途，但普通的一步一更新循环需要每一步独立的梯度。
因此本课明确执行：

```python
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 不把旧缓冲区逐元素写成 0，而是把梯度引用设为 `None`。
下一次 backward 会创建新梯度；这通常更节省无用写入，也能区分“真实的零梯度”
和“本轮根本没有梯度”。

## 6. 梯度 L2 范数

假设所有参数梯度摊平成一个向量 `g`，全局 L2 范数为：

```text
||g||₂ = sqrt(Σᵢ gᵢ²)
```

它把成千上万个梯度压缩为一个可观察数字：

- `0` 可能表示没有梯度路径、模型已处于平坦点，或数据/实现有问题；
- 有限正数说明 backward 产生了信号；
- `NaN` 或 `Inf` 表示数值已经失效，不能继续更新；
- 极大的值可能导致一步跨得太远。

[`gradient_l2_norm`](../../training/nanogpt_nspire/training_loop.py)
同时处理 dense 和 sparse gradient，并使用 float64 累加平方和，减少测量本身的
舍入误差。

## 7. gradient clipping

本课默认设置：

```text
max_grad_norm = 1.0
```

若原始范数超过阈值，PyTorch 近似把所有梯度乘以同一个比例：

```text
g_clipped = g * max_norm / ||g||₂
```

这样保留梯度方向，但限制长度。我们分别记录：

```text
gradient_l2_norm_before_clip
gradient_l2_norm_after_clip
```

clipping 不是修复 NaN 的工具。因此代码先检查裁剪前范数有限，再执行裁剪，
随后再次检查。

## 8. optimizer.step 做什么

本课继续使用 AdamW，但设置 `weight_decay=0.0`，避免在教学实验中混入权重衰减。
Adam 类优化器不会简单地使用：

```text
θ <- θ - learning_rate * gradient
```

它还为每个参数维护梯度的一阶与二阶移动统计。概念上：

```text
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²
θₜ = θₜ₋₁ - learning_rate * corrected(mₜ) / (sqrt(corrected(vₜ)) + ε)
```

因此 checkpoint 若要继续训练，通常还需要 optimizer state；而只做推理时只需
最终模型参数。本项目当前 checkpoint 用于推理和复核，不声称支持无损续训。

## 9. 参数更新范数

有梯度并不自动证明参数发生了变化。例如 learning rate 为 0、优化器参数列表
错误，都会让 `.grad` 存在但模型不更新。

本课在 `optimizer.step()` 前复制参数 `θ_before`，更新后计算：

```text
||θ_after - θ_before||₂
```

记录为：

```text
parameter_update_l2_norm
```

它应当是有限正数。这个复制操作会增加一些训练开销，但模型只有数万个参数，
而本课优先选择可观察性。未来正式训练较大 teacher 时，不会每一步复制全部参数。

## 10. token accuracy

loss 使用整个概率分布；accuracy 只检查最大 logit 对应的 token：

```python
predicted = logits.argmax(dim=-1)
accuracy = mean(predicted == targets)
```

两者表达的信息不同：

- loss 从正确字符概率 `0.1 -> 0.4` 时会明显改善，即使 argmax 仍然错误；
- accuracy 只有预测第一名改变时才跳变；
- fixed batch accuracy 达到 100% 不代表概率足够集中，loss 仍可能继续下降。

所以训练记录同时保留 loss 和 accuracy，但验证集 loss/BPC 仍是主要质量指标。

## 11. train、eval 与 inference_mode

`model.train()` 和 `model.eval()` 控制模块行为。例如以后加入的 dropout 和
LayerNorm/BatchNorm 类模块可能在训练和推理时行为不同。

本课模型还没有 dropout，两个模式的数值暂时相同，但训练工具仍严格管理模式：

- `train_step` 主动进入 train mode；
- `evaluate_batch` 保存原模式，临时进入 eval mode；
- 评估放在 `torch.inference_mode()` 中，不创建计算图；
- 评估结束后恢复调用者原来的模式。

提前建立这个约束可以避免以后加 dropout 后，验证 loss 因模式错误而不可信。

## 12. 一次完整更新

核心实现在
[`train_step`](../../training/nanogpt_nspire/training_loop.py)：

```python
model.train()
optimizer.zero_grad(set_to_none=True)

logits, loss = model(inputs, targets)
check_loss_is_finite(loss)

loss.backward()
measure_gradient_norm_before_clip()
clip_grad_norm_if_requested()
measure_gradient_norm_after_clip()

copy_parameters_before_update()
optimizer.step()
measure_parameter_update_norm()
```

顺序很重要：

- 先清梯度，否则混入上一步；
- backward 前不能丢掉计算图；
- clipping 必须发生在 backward 后、step 前；
- 参数位移必须比较 step 前后；
- 非有限 loss/gradient 必须阻止 optimizer 更新。

## 13. 为什么“能过拟合”是一个测试

如果一个有足够容量的模型连 32 个固定标签都无法记住，优先怀疑：

1. input/target 移位错误；
2. loss 没有连接到模型参数；
3. 忘记调用 backward 或 optimizer.step；
4. 优化器没有拿到正确参数；
5. learning rate 太小、太大或梯度被错误清除；
6. 模型容量不足；
7. 同一个输入对应互相矛盾的标签。

反过来，成功过拟合只关闭“基本优化接线”这一道门。它不证明：

- 模型在未见文本上有效；
- 模型没有数据泄漏；
- 超参数适合正式训练；
- 生成文本有意义；
- 模型适合 Nspire 部署。

## 14. training loss、validation loss 和 generalization gap

本课记录：

```text
fixed_batch_loss
validation_loss
generalization_gap = validation_loss - fixed_batch_loss
```

当 fixed batch 被记住时：

```text
fixed_batch_loss -> 接近 0
```

但验证集包含大量未见窗口，因此 validation loss 通常不会同步接近 0。
正的 generalization gap 在本实验中是预期现象：它恰好显示“记住训练例子”
和“学到可迁移规律”不是一回事。

注意，generalization gap 的绝对大小还受 fixed batch 的特殊性、模型容量和
验证采样影响。本课不把它当作正式模型比较分数。

## 15. 代码和数据流

训练工具：

- [`training_loop.py`](../../training/nanogpt_nspire/training_loop.py)：
  模型无关的单步更新、固定 batch 评估与过拟合循环；
- [`lesson04_overfit.py`](../../training/nanogpt_nspire/lesson04_overfit.py)：
  数据加载、固定 batch 选择、验证集评估、checkpoint 和实验记录。

数据流：

```text
Lesson 01 validated train.bin
  -> seeded make_batch exactly once
  -> fixed inputs/targets
  -> Lesson 03 attention model
  -> repeated observable train_step
  -> fixed-batch metrics
  -> independent validation windows
  -> checkpoint + run.json
```

## 16. 测试证据

训练循环测试覆盖：

- 裁剪前后梯度范数；
- 手工复算当前 `.grad` 的全局范数；
- 参数确实发生更新；
- token accuracy 范围；
- train/eval 模式恢复；
- 无梯度评估；
- 非法 clipping 阈值；
- 非有限 loss 在更新前失败；
- 错误模型返回协议；
- toy model 把固定 batch 拟合到 loss `< 0.01`、accuracy `100%`；
- step 0、step 1、间隔点和最终点的历史记录。

Lesson 04 CLI 测试还会在 CPU 上准备小数据集，验证 checkpoint、`run.json`、
数据哈希、source commit、固定 batch 和指标结构。

当前全套测试为 `61 passed`。

## 17. 真实实验命令

实现提交后使用：

```powershell
python -m nanogpt_nspire.lesson04_overfit `
  --data-dir artifacts/data/tinyshakespeare `
  --output-dir artifacts/lesson04 `
  --device auto `
  --seed 1337 `
  --steps 1000 `
  --batch-size 1 `
  --block-size 32 `
  --embedding-dim 64 `
  --learning-rate 0.01 `
  --max-grad-norm 1.0 `
  --record-every 100 `
  --eval-batches 50 `
  --target-training-loss 0.05 `
  --source-commit c9efdafe1a93427246ffab5b2194c4b9576ccd80
```

生成：

```text
artifacts/lesson04/
├── overfit_attention_lm.pt
└── run.json
```

两个文件都由 Git 忽略。可长期复核的小型结果摘要会进入 `experiments/`。

## 18. 真实实验结果

实验使用提交 `c9efdaf`，结果保存在
[`lesson04-overfit.json`](../../experiments/lesson04-overfit.json)。
固定 batch 解码为：

```text
h of mankind
Would hang themselve
```

它是训练集中由 seed `1338` 选中的连续 33 个字符：前 32 个构成 input，
后 32 个构成右移一位的 target。

### 过拟合结果

| 指标 | 训练前 | 1000 步后 |
|---|---:|---:|
| fixed-batch loss | 4.470628 | 0.000008911 |
| fixed-batch BPC | 6.449753 | 0.000012856 |
| fixed-batch token accuracy | 0% | 100% |
| validation loss | 4.433673 | 17.774596 |
| validation BPC | 6.396438 | 25.643322 |

预先规定的成功门槛是：

```text
final_fixed_batch_loss <= 0.05
```

最终 loss 为 `8.91×10⁻⁶`，降低 `99.9998%`，门槛通过。到第一个间隔记录点
step 100 时，loss 已为 `7.52×10⁻⁵` 且 accuracy 已达 100%。由于我们只每
100 步记录一次，不能从该记录判断首次达到 100% 的精确 step。

### 梯度和参数更新

| step | clip 前梯度范数 | clip 后梯度范数 | 单步参数更新范数 |
|---:|---:|---:|---:|
| 1 | 2.111147 | 1.000000 | 1.542897 |
| 100 | 0.000363 | 0.000363 | 0.002499 |
| 500 | 0.000090 | 0.000090 | 0.001419 |
| 1000 | 0.000038 | 0.000038 | 0.000977 |

第一步梯度超过阈值并被裁剪到约 1.0。随着正确字符概率接近 1，loss 和梯度信号
同时变小，但 step 1000 的参数更新仍为有限正数。初始到最终参数向量的总 L2
位移为 `13.983730`。

### 这次“失败”正是实验成功

最终 generalization gap 为：

```text
17.774596 - 0.000008911 = 17.774588
```

validation loss 相比训练前反而增加 `13.340923`。模型用 26,752 个参数记住了
32 个 target，并把概率分布推得非常尖锐；这些特化参数在随机 validation
窗口上给错误字符很低概率，于是交叉熵大幅恶化。

因此本实验同时给出两条证据：

1. forward、loss、backward、clipping 和 optimizer update 确实连通；
2. 最小化训练例子的 loss 不等于学习可泛化的语言规律。

这个 checkpoint 是故意制造的反例，不应作为 Lesson 03 正常 checkpoint 的
升级版，也不进入后续量化质量比较。

### 性能边界

1000 步共处理 32,000 token，CUDA 计时 `6.073 s`，表观吞吐量约
`5,269 token/s`。这个数字远低于 Lesson 03，主要因为本课每一步都同步读取
梯度范数，并复制参数计算更新范数；它是可观察性成本，不能拿来比较模型训练
速度，更不能推断 Nspire 推理性能。

### 独立复核

- checkpoint：`112,332 bytes`；
- checkpoint SHA-256：
  `b4b8426eaec4a24ca16c62aa95c9334cc0ea479a6e8ff19e463f4590b3216245`；
- 严格回载 7 个 state-dict tensor，无 missing/unexpected key；
- checkpoint 中的 fixed inputs/targets 与 `run.json` 完全一致；
- CUDA 复算 fixed-batch loss 和 100% accuracy，与记录逐位一致；
- 固定 validation windows 复算 loss `17.774596443176268`，逐位一致；
- 所有记录点的 loss、梯度范数和参数更新范数均有限；
- clip 后梯度范数均不超过 `1.0 + 10⁻⁶`；
- checkpoint 与 `run.json` 均由 Git 忽略。

## 19. 下一课：量化

完成过拟合接线检查后，Lesson 05 将开始区分：

- 训练时的 FP32 参数；
- 推理时的量化整数；
- scale、zero point 和量化误差；
- 只压缩文件但运行时解压为 FP32；
- 真正由 C runtime 直接消费 INT8/INT4。

量化实验仍会保留 FP32 reference path，并比较 logits、loss、文件大小与内存，
不会只报告“模型文件变小了”。
