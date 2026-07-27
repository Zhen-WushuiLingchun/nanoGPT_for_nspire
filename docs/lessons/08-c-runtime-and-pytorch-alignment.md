# Lesson 08：统一模型文件、C 推理与 PyTorch 对齐

这一课把前三条路线从“PyTorch 里可以运行”推进到了“同一份模型文件可以被纯 C
直接推理”：

```text
Direct-Small     FP32 .ngm -> Host C
Distilled-Small  FP32 .ngm -> Host C
Quantized-Small  packed INT4 .ngm -> W4A8 Host C
                                     |
                                     +-> Ndless ARM 编译、链接、封装
```

这里最重要的结果不是生成了几句 Shakespeare，而是四个部署条件同时成立：

1. 三个模型使用同一个有界、带校验和的二进制格式；
2. FP32 C 与 PyTorch 的固定提示 logits 对齐；
3. Quantized-Small 直接读取 packed INT4，没有先还原整份 FP32 权重；
4. 完整 portable runtime 已被原生 Ndless 工具链编译进 `.tns`。

真机运行速度、真机峰值 heap 和最终对话界面仍属于下一课。本课的 ARM 证据是
compile/link/package，不冒充 CX II 实测。

## 1. 先看最终结果

固定提示为：

```text
First Citizen:
```

比较“提示最后一个位置的 logits”，再从同一状态贪心生成 64 个字符：

| 路线 | `.ngm` bytes | arena bytes | max abs | RMSE | 64 token |
|---|---:|---:|---:|---:|---|
| Direct-Small | 5,046,592 | 663,232 | 3.81e-6 | 1.55e-6 | 完全一致 |
| Distilled-Small | 5,046,592 | 663,232 | 4.77e-6 | 1.49e-6 | 完全一致 |
| Quantized-Small W4A8 | 6,036,544 | 2,378,624 | 2.86e-6 | 6.69e-7 | 完全一致 |

预注册门为：

```text
max absolute error <= 2e-4
RMSE               <= 5e-5
64-token greedy    必须逐 token 完全一致
```

三条路线全部通过。

Host C 的一次标量实现测得：

| 路线 | forward tokens/s |
|---|---:|
| Direct-Small | 约 2,108 |
| Distilled-Small | 约 2,108 |
| Quantized-Small W4A8 | 约 133 |

这不是说 INT4 天生比 FP32 慢。当前 FP32 可被编译器很好地优化，而 W4A8 kernel
仍在每次点积内逐 nibble 解码，没有 SIMD、行分块或 ARM 专用优化。本课优先证明
算术和内存语义正确；这个明显的速度差正是下一课的优化基线。

## 2. 为什么不能直接让 C 读取 `.pt`

PyTorch checkpoint 适合训练恢复，里面有 Python/PyTorch 的序列化结构、tensor
metadata，甚至可能包含 optimizer state。让嵌入式 C 解析它会带来：

- 格式复杂且不是本项目控制的稳定 ABI；
- 需要实现大量与推理无关的反序列化逻辑；
- 很难在读入前证明文件和内存上限；
- tied weight 容易被重复保存；
- INT4 package 仍可能被不小心展开成 FP32。

因此我们定义 `.ngm`（nanoGPT Nspire Model）：

```text
128-byte header
64-byte × tensor_count tensor table
UTF-8 character vocabulary
64-byte aligned tensor payloads
```

header 使用 magic：

```text
NGNSP001
```

并记录：

- schema version 与 little-endian 标记；
- vocab、context、layer、head、width、MLP ratio；
- FP32 或 W4A8 路线；
- group size 与 activation quantization；
- 文件总长度和各 section offset；
- header CRC32 与 payload CRC32。

C loader 在暴露任何 tensor pointer 前，会验证长度、整数溢出、offset、对齐、重叠、
shape、storage、CRC、UTF-8 词表和总内存上限。损坏的文件应当被拒绝，而不是让错误
offset 变成任意内存读取。

## 3. 为什么 tensor 使用数字 ID

训练时用名字很方便：

```text
blocks.2.attention.qkv.weight
```

设备推理若每次做字符串查找，会增加代码、内存和出错面。因此格式 v1 冻结数字 ID：

```text
1       token embedding / tied lm_head
2       position embedding
100+    block 0
110+    block 1
...
1000    final LayerNorm weight
```

每个 block 固定六项：

```text
slot 0 attention LayerNorm
slot 1 fused QKV matrix
slot 2 attention output matrix
slot 3 MLP LayerNorm
slot 4 MLP input matrix
slot 5 MLP output matrix
```

以 embedding width `C`、MLP ratio `r` 表示，shape 为：

| tensor | shape |
|---|---|
| token embedding | `(V, C)` |
| position embedding | `(T, C)` |
| QKV | `(3C, C)` |
| attention output | `(C, C)` |
| MLP input | `(rC, C)` |
| MLP output | `(C, rC)` |
| LayerNorm weight | `(C,)` |

二维权重按 row-major 保存。`nn.Linear(C_in, C_out)` 的 weight 本来就是
`(C_out, C_in)`，所以 C 的一行点积正好产生一个 output 分量：

```text
y[row] = sum(weight[row, column] * x[column])
```

## 4. tied lm_head 为什么只存一次

三个模型都满足：

```text
lm_head.weight is token_embedding.weight
```

输入时，第 `token_id` 行是该 token 的 embedding；输出时，同一矩阵的每一行与
最终 hidden state 做点积，得到 vocabulary logits。

`.ngm` 只保存 tensor ID 1 一次。C 端既用它做 lookup，也用它做最后的矩阵乘法。
Direct/Distilled 因此各有 27 个物理 tensor，而不是人为多出一份 head。

## 5. 一个 token 如何穿过完整 GPT

推理状态先取：

```text
x = token_embedding[token] + position_embedding[position]
```

每个 pre-norm block 执行：

```text
n = LayerNorm(x)
q, k, v = split(W_qkv @ n)
cache.append(k, v)
a = causal_attention(q, cached_k, cached_v)
x = x + W_attn_out @ a

n = LayerNorm(x)
m = GELU(W_mlp_in @ n)
x = x + W_mlp_out @ m
```

最后：

```text
n = final_LayerNorm(x)
logits = token_embedding @ n
```

推理阶段 dropout 已关闭，所以 C 不需要随机 dropout。训练时 dropout 改变了学到的
参数，这正是 Lesson 07 中 Teacher v1/v2 不同的来源；推理图本身没有 dropout
算子。

## 6. LayerNorm、softmax 与 GELU

### LayerNorm

对一个宽度为 `C` 的 token vector：

```text
mean = sum(x) / C
var  = sum((x - mean)^2) / C
y_i  = (x_i - mean) / sqrt(var + 1e-5) * weight_i
```

模型没有 LayerNorm bias，因此只需一个 weight vector。这里的方差分母是 `C`，
不是样本方差的 `C-1`。

### stable softmax

直接计算 `exp(score)` 可能溢出。先减最大值：

```text
p_i = exp(score_i - max(score))
      / sum_j exp(score_j - max(score))
```

减去同一个常数不会改变 softmax 概率，却把最大指数固定为 `exp(0)=1`。

### tanh GELU

PyTorch 模型冻结为：

```text
GELU(approximate="tanh")
```

C 必须用同一近似：

```text
0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
```

如果 C 改用精确 `erf` GELU，即使数学上也合理，它也不再是同一个模型。

## 7. KV cache 为什么能让连续对话可行

若生成第 `t` 个 token 时重新计算前面 `t` 个 token，生成成本会随上下文不断增长。
causal attention 的旧 token key/value 在后续不会改变，所以可以保存：

```text
key_cache   [layer, position, embedding]
value_cache [layer, position, embedding]
```

每来一个新 token，只计算它自己的 QKV，把新 K/V 写入当前 position，再让 Q 查询
`0..position` 的 cache。

本项目不在 `ng_runtime_forward_token` 内分配内存。arena 在启动时一次性切成：

```text
K/V cache
hidden and normalized
fused QKV
attention context
projection
MLP hidden
logits
one-head attention scores
W4A8 activation scratch, if needed
```

attention heads 串行计算，因此 scores 只需 `T` 个 float，而不是
`n_head*T`。这把 Direct/Distilled arena 收紧为 `663,232 bytes`。

## 8. FP32 对齐到底比较什么

Python 从 `.ngm` 重新构造 evaluation-mode 模型，而不是继续信任原 `.pt`。
这同时验证导出后的 tensor 顺序和值。

对同一提示：

1. PyTorch 得到提示最后位置的 vocabulary logits；
2. C 得到同一位置 logits；
3. 计算 max absolute error 与 RMSE；
4. 两边各自贪心选 64 次 argmax；
5. 要求每个生成 token 完全相同。

logits 是连续值，允许很小的浮点 reduction 差异；token 序列是离散行为，不能“差
一点”。二者结合比只看一句生成文本更有诊断力。

## 9. 真正的 W4A8，而不是启动时偷解包

Lesson 07 的 PyTorch INT4 质量参考采用：

```text
packed W4 -> 重建完整 FP32 matrix -> FP32 activation
```

它应称为 W4A32 reference，能测 weight rounding，却不满足设备内存约束。

本课的矩阵 kernel 对每个 input group 执行：

```text
a_scale = max(abs(a_group)) / 127
q_a     = round_to_even(a_group / a_scale)
q_a     = clamp(q_a, -127, 127)

int_dot = sum(q_w * q_a)              # INT32
output += int_dot * w_scale * a_scale # FP32 rescale
```

全零 activation group 使用 `scale=1, q=0`。权重 nibble 在点积循环中 low-first
解码，取值仍为 `[-7,7]`。activation rounding 明确使用 ties-to-even，与
`torch.round` 对齐。

内存里存在的是：

```text
6,036,544-byte packed model blob
2,378,624-byte KV/scratch arena
```

合计：

```text
8,415,168 bytes
```

没有约 42.8 MB 的 Teacher FP32 matrix 副本，也没有 parameter-sized FP32 scratch。
因此它满足三路线比较中“真正整数推理、不能启动后偷偷还原 FP32”的定义。

## 10. A8 会不会进一步伤害质量

使用与 Lesson 07 相同的 50 个 validation batches、batch size 64、context 128 和
seed 1338：

| reference | validation loss | BPC |
|---|---:|---:|
| W4A32 dequantized | 1.4737991 | 2.1262427 |
| packed W4A8 | 1.4737427 | 2.1261613 |
| Direct-Small FP32 | 1.4997900 | 2.1637396 |

W4A8 相对 W4A32 的差是：

```text
-0.0000564
```

它通过“额外退化不超过 `0.02`”以及“仍优于 Direct-Small”两个门。负号不能解释成
“INT8 activation 提升模型”；这个差只有约 `5.6e-5`，来自不同 rounding 路径的
微小扰动。可信结论是：

```text
在冻结窗口上没有观察到有意义的额外质量损失。
```

## 11. 三条路线现在能公平比较到哪一步

| 路线 | 方法 | 参数 | 部署文件 | loss | Host C |
|---|---|---:|---:|---:|---|
| Direct-Small | 随机初始化小架构直接训练 | 1,261,120 | 5,046,592 B | 1.499790 | 完成 |
| Quantized-Small | Teacher v2 后训练 W4A8 | 10,695,936 | 6,036,544 B | **1.473743** | 完成 |
| Distilled-Small | 同小架构学习 hard+soft 目标 | 1,261,120 | 5,046,592 B | 1.522163 | 完成 |

现在 Quantized-Small 的 `integer_runtime_required` 已不再 pending。它在同一文件预算
内用低精度容纳更多参数，验证质量最好；但当前朴素 Host kernel 明显最慢。

Direct 与 Distilled 的同架构结论不变：基础蒸馏没有胜过直接训练。C runtime
不会改变模型 loss，只证明两个 artifact 能以同一方式部署。

## 12. Ndless 原生构建已经验证了什么

使用已安装的 Ndless SDK r2022 与 Arm GNU Toolchain 14.3：

```text
nspire-gcc -> ARM ELF -> genzehn -> make-prg -> .tns
```

portable loader、FP32 operators、W4A8 operators 和增量 runtime 都使用：

```text
-std=c11 -marm -Os -Wall -Wextra -Werror
```

原生 smoke package：

```text
dist/nanogpt-runtime-smoke.tns  48,056 bytes
ARM text                         58,356 bytes
ARM data                          4,416 bytes
ARM bss                             868 bytes
```

smoke 入口会实际链接 runtime，并检查初始化、清空和缺 tensor 的受控错误路径。

它没有证明：

- `.tns` 已在用户的 CX II 上成功启动；
- 8.4 MB model+arena 等于真机峰值 heap；
- Host 的 133/2108 tokens/s 可以外推到 ARM；
- 320×240 UI 已完成。

这些边界会原样带入下一课。

## 13. 为什么退出清理现在就进入 runtime

`ng_runtime_reset` 不只把 `position` 设成零，还会用 observable volatile stores
覆盖整个 arena。arena 包含：

- 所有历史 K/V；
- 当前 hidden、MLP 和 logits；
- W4A8 临时 activation。

这样“New Chat”和“Exit”可以共享一个明确的隐私原语。下一课的 UI 仍需保证：

1. 覆盖输入框与 transcript buffer；
2. 调用 `ng_runtime_reset`；
3. 释放 arena；
4. 释放 model blob；
5. 恢复 Ndless framebuffer/input 状态；
6. 不把对话写入文件。

模型权重本身不是对话数据，可以重新从 `.ngm` 加载。

## 14. 下一课：原生像素对话界面和真机测量

当前 Tiny Shakespeare 模型仍是文本续写器，没有经过 instruction/chat 后训练。
所以界面可以先长得像聊天软件，但 AI 的语义还不是 ChatGPT。后续做物理解释数据
后训练，才会逐步改变这一点。

下一课会实现：

- 连续 USER/AI cell；
- 输入、提交、生成中断与滚动；
- 逐 token 刷新；
- time-to-first-token、tokens/s、context、内存状态；
- New Chat 清 cache；
- Exit 清对话且不持久化；
- CX II 上的模型加载、峰值内存与速度实测。

界面设计已冻结在
[`2026-07-28-lesson-09-ndless-chat-ui-design.md`](../plans/2026-07-28-lesson-09-ndless-chat-ui-design.md)。

## 15. 复现实验

Host：

```powershell
cmake -S . -B build/host -G "Visual Studio 17 2022" -A x64
cmake --build build/host --config Release
ctest --test-dir build/host -C Release --output-on-failure

python -m nanogpt_nspire.alignment `
  --runner build/host/Release/run_model.exe `
  --model artifacts/lesson08-export/direct-small.ngm `
  --prompt "First Citizen:" `
  --generate 64 `
  --output artifacts/lesson08-export/direct-fp32-alignment.json
```

Ndless：

```bash
eval "$(bash tools/ndless-env.sh)"
make ndless-smoke
```

机器可读摘要：

- [`lesson08-c-runtime.json`](../../experiments/lesson08-c-runtime.json)
- [`small-model-comparison.json`](../../experiments/small-model-comparison.json)
