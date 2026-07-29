# Lesson 18：把可控 CoT 模型真正放上 Nspire

Lesson 16 得到了一个 `6-layer / width 384 / GQA 6:2 / ALiBi / context 512`
的 SFT v2 checkpoint。它在冻结评测中能 100% 遵守 Direct/Think 格式并正常
结束，但 Direct/Think 都只答对 `2/128`。Lesson 17 的三条 RL 路线没有同时
改善 Primary 与 Challenge，也没有通过格式门，因此本课不事后挑一个最好 seed
上机，而是部署 Lesson 16 SFT v2，研究以下问题：

1. PyTorch 的 byte tokenizer、GQA、ALiBi 和 512 context 能否导出为真实 W4A8？
2. Host C 是否与 packed PyTorch reference 对齐？
3. Nspire 界面能否真正用模型可见的 `<THINK>` / `<FINAL>` 切换两种模式？
4. 文件能否通过 CLI 完整传输并回读校验？
5. “有 CoT”是否等于“会正确推理”？

机器可读摘要位于
[`lesson18-gqa-alibi-cot-device-pilot.json`](../../experiments/lesson18-gqa-alibi-cot-device-pilot.json)。

## 1. 先回答：它现在会流利英语和 CoT 了吗

准确答案是：

```text
它会生成局部通顺的英语句子。
它会按训练协议进入 THINK，再用 FINAL 切回最终答案。
它还不是流利、开放域、可靠的英语助手。
它生成的 reasoning 也不等于正确的 reasoning。
```

Lesson 16 已经给出最重要的冻结证据：

| mode | format/mode | task exact | challenge exact |
|---|---:|---:|---:|
| Direct | 100% | 2/128 | 1/256 |
| Think | 100% | 2/128 | 1/256 |

因此“有 CoT”的严格表述应是：

> 模型学会了可控 reasoning 输出格式，而不是学会了可靠的数学推理算法。

## 2. 这次 USER、ASSISTANT、THINK 不再只是界面标签

Lesson 09 的 Tiny Shakespeare 模型没有 role token。屏幕上的 `USER` 和 `AI`
只是 UI metadata，模型看到的仍是一串字符续写。

本课使用 Lesson 10 冻结的 264-token 词表：

```text
0..255: raw bytes
256: BOS
257: EOS
258: USER
259: ASSISTANT
260: TOOL
261: THINK
262: FINAL
263: PAD
```

第一轮 Direct prompt 是：

```text
<BOS><USER>question bytes<ASSISTANT><FINAL>
```

第一轮 Think prompt 是：

```text
<BOS><USER>question bytes<ASSISTANT><THINK>
```

Think 模式中，运行时把 `<THINK>` 之后、`<FINAL>` 之前的 byte token 放入独立
`THINK` cell；收到 `<FINAL>` 后创建 `AI` cell；收到 `<EOS>` 后停止。也就是说，
两种模式由训练时模型可见的控制 token 驱动，而不是把同一串 completion 换一种
颜色显示。

## 3. `.ngm v2`：GQA、ALiBi 与 byte-special tokenizer

旧 `.ngm v1` 只支持 learned position embedding、普通 MHA、文件内字符词表和
`block_size <= 128`。本课保留 v1 兼容，同时给 v2 header 增加：

- `n_kv_head`；
- `position_mode = ALiBi`；
- `tokenizer_type = byte + fixed special tokens`；
- `block_size <= 512`。

导出器直接从 Lesson 16 checkpoint 写出 groupwise packed INT4 matrix、FP32
RMSNorm 和动态 INT8 activation 所需 scale。C 运行时不会在启动后把矩阵展开
成 FP32。部署模型为：

| field | value |
|---|---:|
| route | GQA-ALiBi-SFT-v2-Context512 |
| layers / Q heads / K/V heads | 6 / 6 / 2 |
| width / context / vocabulary | 384 / 512 / 264 |
| source FP32 checkpoint | 38,592,622 B |
| packed W4A8 `.ngm` | 5,387,968 B |
| model SHA-256 | `91d52d9d693059fb38f6f2c151bf470d9027b770b467dc11c50ec2106d9beadf` |
| runtime arena | 3,165,312 B |
| model + arena lower bound | 8,553,280 B = 8.16 MiB |

最后一项不包括 framebuffer、chat cells、程序映像和 allocator overhead，因此
不是 CX II 真实峰值 RAM。真实峰值仍必须由真机运行时计数器测量。

## 4. C 运行时怎样实现 GQA 与 ALiBi

GQA 的 6 个 query head 共享 2 组 K/V：

```text
query head 0,1,2 -> KV head 0
query head 3,4,5 -> KV head 1
```

因此 512 context 的 FP32 KV cache 理论值是：

```text
2 * layers * n_kv_head * context * head_dim * sizeof(float)
= 2 * 6 * 2 * 512 * 64 * 4
= 3,145,728 bytes
```

若仍使用 6-head MHA，同一 cache 会是 9 MiB。GQA 在这里主要节省 KV RAM，
不会减少 MLP matrix 的计算。

ALiBi 不保存 512 行 learned position embedding，而是在每个 attention score
上加入与相对距离有关的 head-specific linear bias。Host C 与 PyTorch reference
使用相同 slope 构造。

## 5. 对齐结果：短序列严格通过，长 CoT 序列贪心一致

Direct prompt `Hello`，生成 32 token：

| check | value |
|---|---:|
| greedy token sequence | exact |
| max absolute logits error | `1.907e-6` |
| RMSE | `3.807e-7` |
| strict gate | pass |

Think prompt `What is 12 times 7?`，生成 96 token：

| check | value |
|---|---:|
| greedy token sequence | exact |
| max absolute logits error | `0.031934` |
| RMSE | `0.013266` |
| old strict FP32-style logit gate | fail |

第二条不能写成“全部 logits 对齐通过”。随着 incremental prefix 变长，标量 C
和 PyTorch packed reference 的浮点累积顺序产生了可见偏差；这次 96-token
贪心序列仍逐 token 相同，但旧的 `2e-4 / 5e-5` gate 不适合被悄悄放宽。
后续应增加按位置绘制误差与首个 divergence 的长序列门，而不是只报最后一个
最大值。

## 6. Host 输出暴露的真实能力边界

Direct：

```text
[USER]
Hello
[AI]
The answer is 10000.
```

Think：

```text
[USER]
What is 12 times 7?
[THINK]
Formula F = m a. Substitute F = 12 * 7 = 12 N.
[AI]
The force is 12 N.
```

第二个例子非常重要。它正确进入 Think cell、写出像公式代入的英语，并正确产生
`<FINAL>` 和 `<EOS>`；reasoning 与 final 也局部一致，但乘法、变量绑定、
物理公式和单位都错了。

同一 prompt 的 FP32 checkpoint 与 W4A8 导出得到相同正文，因此这个失败不是
INT4 量化凭空造成的。它延续 Lesson 16 的结论：

```text
format compliance != reasoning correctness
fluent-looking explanation != grounded calculation
CoT token budget != CoT capability
```

### 6.1 `10/100/1000/10000` 是模型吸引盆，不是固定 prompt

真机首次运行后，用户观察到不同问题经常都回答 `The answer is 10000.`。为了
区分“前端又给所有问题喂了同一串 token”和“模型本身塌缩到高频答案”，Host C
用设备上完全相同的 W4A8 文件做了额外 prompt sweep：

| prompt | Direct completion |
|---|---|
| `hello` | `The answer is 10000.` |
| `what is 1 plus 1?` | `The answer is 10.` |
| `WHAT IS 1 PLUS 1?` | `The power is 20 W.` |
| `What is 12 times 7?` | `The answer is 100.` |
| `Explain gravity.` | `The answer is 1000.` |
| `do not say 10000` | `The answer is 100000.` |

输入内容和大小写都会改变输出，C 单元测试也逐 token 检查了 prompt 中的 user
byte。因此目前证据不支持“前端固定忽略了用户输入”。更符合证据的解释是：

- SFT 语料很窄，大量 Direct target 都是 `The answer is <number>.`；
- 开放域聊天 prompt 落在训练分布外；
- 小模型用 `10` 的幂和少量物理单位形成高概率吸引盆；
- 多轮时，上一轮错误 completion 仍在 context 中，会进一步强化同一模式；
- 否定式提示 `do not say 10000` 不是硬约束，反而把 `10000` byte pattern 放入
  上下文。

真机诊断应先按 `Menu` 创建 New Chat，再输入完全相同的
`what is 1 plus 1?`。若仍与 Host 的 `The answer is 10.` 不同，才进入真机
token trace 调试；不能仅凭多个分布外问题都落入 `10000` 就判定 UI 回归。

## 7. 真机界面

顶部状态栏显示：

```text
NANOGPT | used/512 | GQA W4 | DIRECT or THINK
```

控制方式：

| key | action |
|---|---|
| `Tab` | 在 DIRECT / THINK 间切换 |
| `Enter` | 发送 |
| `Del` | 删除输入字符 |
| `Up/Down` | 滚动 |
| `Menu` | New Chat，清空上下文但保留当前模式 |
| `Esc` | 取消当前生成 |
| `Ctrl+Esc` | 退出并清空会话内存，不保存对话 |

Direct 输入行显示 `D>`，Think 输入行显示 `T>`。Think reasoning 使用独立、
弱化颜色的 cell，最终答案仍显示为 `AI` cell。默认最大生成长度由 96 提高到
256 token，但实际生成仍受 512 context 共同限制。

## 8. CLI 部署与传输现象

设备端使用独立目录：

```text
/nanoGPT/nanogpt-chat.tns
/nanoGPT/model.ngm.tns
```

最终文件：

| file | bytes | SHA-256 |
|---|---:|---|
| `nanogpt-chat.tns` | 62,310 | `e0581d074e71db78a9af3c428be1e60b19686fc21aa0d94b6a9c55de601570b6` |
| `model.ngm.tns` | 5,387,968 | `91d52d9d693059fb38f6f2c151bf470d9027b770b467dc11c50ec2106d9beadf` |

一次双文件 sync 中，程序上传、SHA-256 回读、替换和删除旧副本成功；模型完整
上传并通过 size gate，但第一次 SHA-256 回读到约 90% 时 USB/IP 会话断开。
重新附加同一个 USB 后，CLI 用 `--reuse-temporary` 复用已经上传的
`model.ngm.upload.tns`，完整读回 5,387,968 B，SHA-256 与本地一致，再原子替换
旧模型并删除 rollback。最终目录核验只有两个正式文件，没有 `.upload` 或
`.previous`。

这里证明的是：

```text
Windows detection -> usbipd/WSL attach -> upload -> full readback hash
```

它证明传输字节完整，不等于证明计算器已经成功启动新程序、完成一次生成或达到
某个速度/RAM 指标。后者需要真机打开应用后的屏幕和测量证据。

## 9. 下一步真机复测协议

先固定 greedy decode，避免把采样随机性混进移植问题：

1. 打开 `/nanoGPT/nanogpt-chat.tns`，记录是否成功加载 `GQA W4`；
2. Direct 输入 `Hello`，确认是否得到 `The answer is 10000.`；
3. `Tab` 切换 Think，输入 `What is 12 times 7?`；
4. 确认 reasoning 出现在独立 `THINK` cell，final 出现在 `AI` cell；
5. 记录 TTFT、tokens/s、context、tracked RAM；
6. `Menu` 清空会话，确认模式保留；
7. `Ctrl+Esc` 退出，再次打开确认没有保存旧对话。

前两个错误输出是 Host reference，不是期望的正确答案。真机若逐字符一致，说明
部署和 C 路径复现成功；不能因此给模型能力判定为正确。

## 10. 本课声明边界

- 已完成 `.ngm v2`、GQA、ALiBi、512 context、byte-special tokenizer 和
  Direct/Think C chat protocol。
- 已完成 Host build/test、短 Direct 严格 logits 对齐、长 Think 贪心序列对齐、
  Ndless ARM build/package。
- 已完成真机两文件上传、完整 SHA-256 回读、原子替换和无备份目录核验。
- 用户已报告新版本能在 CX II 上启动并做 Direct 生成，但还没有固定 prompt 的
  逐字符对照、Think cell 照片、速度和峰值 RAM 证据。
- 模型有可控 CoT 格式，但冻结准确率和 Host 样例都明确否定“可靠推理助手”。
- 本课没有部署 Lesson 17 的 RL checkpoint，也没有宣称 RL 提升。
