# Lesson 13：外部合成数据 SFT、本地 logit teacher 与严格蒸馏

Lesson 12 的 10.8M student 已经学会 `<USER>` / `<ASSISTANT>` 格式，却只在冻结的
128 题中答对 2 题。本课不把“换一个更强 API 制备合成答案”和“逐 token 蒸馏
概率”混成一个模糊的 teacher 实验，而是拆成两条路线：

```text
External V4-Pro synthetic-data generator
  -> verified answer text
  -> hard-label SFT

Local 59.3M GPT with the same tokenizer
  -> 264-dimensional logits at every position
  -> strict temperature-scaled KL distillation
```

第一条教 student “应该输出哪段文字”；第二条还会教它“除 gold byte 外，teacher
认为哪些 byte 也相对合理”。第一条更规范的名称是 **verified synthetic-data
SFT**：它可能传递解题覆盖、语气和用词偏好，但不是严格意义上的蒸馏。只有第二
条让 student 学习 teacher 的条件似然分布，才是本课的 **logit distillation**。
它们的数据、数学目标和可解释结论都不同。

本课新增实现：

- [`secret_safety.py`](../../training/nanogpt_nspire/secret_safety.py)：
  runtime-only credential、错误脱敏与提交物扫描；
- [`external_teacher.py`](../../training/nanogpt_nspire/external_teacher.py)：
  当前 V4-Pro request/response contract、预算和重试；
- [`lesson13_sequence_data.py`](../../training/nanogpt_nspire/lesson13_sequence_data.py)：
  family 选择、exact verifier、quarantine 与固定评测拼接；
- [`local_teacher_train.py`](../../training/nanogpt_nspire/local_teacher_train.py)：
  共享 tokenizer 的 59.3M teacher；
- [`lesson13_distill_train.py`](../../training/nanogpt_nspire/lesson13_distill_train.py)：
  三个新 student route 的同预算训练；
- [`distillation.py`](../../training/nanogpt_nspire/distillation.py)：
  assistant-masked、FP32 loss math 的 CE/KL。

机器可读协议与结果：

- [`lesson13-teacher-distillation.json`](../../experiments/lesson13-teacher-distillation.json)
- [`Lesson 13 implementation plan`](../plans/2026-07-28-lesson-13-teacher-distillation.md)

## 1. 本课的四个 student 对照

所有 student 都从 Lesson 12 的同一个 `Math-Physics-CPT` checkpoint 开始：

```text
SHA-256
ab17a536a58f664f49ff75d176baff7e219996d7d57ce2b6d097eec0b4f89dfb
```

共同冻结：

| 项目 | 固定值 |
|---|---:|
| layers / heads / width | `6 / 6 / 384` |
| parameters | `10,821,504` |
| vocabulary | `264` |
| context | `256` |
| optimizer updates | `1,000` |
| tokens/update | `4,096` |
| sampled tokens | `4,096,000` |
| max/min LR | `1e-4 / 1e-5` |
| frozen generation | 128 prompts，greedy，最多 48 byte tokens |

比较对象：

| Route | 训练数据 | 训练目标 |
|---|---|---|
| `Role-Aware-SFT` | Lesson 12 原答案 | hard CE |
| `Verified-Sequence-SFT` | 原 SFT + 外部模型的已验证合成 sequence | hard CE |
| `Local-Logit-Distilled-SFT` | Lesson 12 原答案 | hard CE + local KL |
| `Combined-Sequence-Logit-SFT` | 外部已验证 sequence | hard CE + local KL |

最后一条是组合增强，只能回答“两个干预一起会怎样”，不能单独证明收益来自
sequence 还是 KL。

## 2. API 合成数据为什么不应称为严格蒸馏

DeepSeek API 使用自己的 tokenizer，返回最终文本；项目拿不到与本地 264-token
词表逐项对齐的完整概率向量。因此：

```text
API text -> 本地重新编码 -> hard target
```

本课把它称为 **verified synthetic-data SFT**。它也可以宽泛地归入
sequence-level knowledge transfer，但不能冒充严格的 knowledge
distillation。即使生成这些文字的模型内部使用了概率分布，我们训练 student 时
也只看到了离散文本 hard targets，没有学习该模型的似然分布。

这条路线最可能直接传递的是：

- 哪些训练问题获得更丰富的正确答案覆盖；
- 解释通常如何组织成短句；
- 常用语气、术语和措辞偏好；
- 在 hard target 中显式出现的中间计算步骤。

因此，如果它胜过 ordinary SFT，严格结论也只是“经过验证的合成数据改善了
student”，不能单凭这个实验说“完成了 logits 蒸馏”。

真正的 logit distillation 必须满足：

```text
teacher vocab index i == student vocab index i
```

所以第二条路线才专门训练一个共享项目 tokenizer 的 local teacher。

## 3. 为什么改用 `deepseek-v4-pro`

旧设计曾写 `deepseek-chat` / `deepseek-reasoner`。到本课执行日
`2026-07-28`，这两个兼容名称已经过官方退役日期。当前官方 API 列出的模型为：

```text
deepseek-v4-flash
deepseek-v4-pro
```

本课固定：

```text
base_url        https://api.deepseek.com
model           deepseek-v4-pro
thinking        enabled
reasoning_effort high
response_format json_object
stream          false
```

官方链接：

- [API quick start](https://api-docs.deepseek.com/quick_start/pricing-details-usd/)
- [Models and pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [Open Platform terms](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html)

开放平台条款明确允许在遵守条款和法律的前提下，将 Output 用于训练其他模型，
包括 model distillation。但“允许训练”不等于“答案一定正确”，所以本课仍执行
独立 verifier。

## 4. API key 为什么不能只靠 `.gitignore`

`.gitignore` 只能阻止未跟踪文件被普通 `git add` 收入，不能解决：

- 把 key 写进源码；
- 把 key 放进 CLI argument 和 shell history；
- provider error 原样回显 authorization header；
- 把 config、request、run manifest 或 checkpoint 序列化；
- 已经被 Git 跟踪的 `.env`。

本课额外实施：

1. key 只在实际 request 前从 `DEEPSEEK_API_KEY` 读取；
2. config 和 CLI 根本没有 `api_key` 参数；
3. dry-run 不读取 key；
4. HTTP 错误只报告 status，不保存 body/header；
5. run artifact 在写入前递归扫描 credential shape；
6. 提交前扫描 tracked tree 与 ignored experiment metadata；
7. `.env`、`.env.*`、`*.credentials`、`*.key`、provider logs 和
   `artifacts/teacher-api/` 全部忽略。

代码和日志可以公开，credential 不属于实验 provenance。

## 5. 先提供 exact ground truth，再让 teacher 教表达

算术和本课的数值物理都能由本地 `Decimal`/公式代码精确算出。向外部合成数据模型
发送的问题同时包含：

```json
{
  "question": "Calculate 12 * 7.",
  "expected_final_answer": "84",
  "expected_unit": null,
  "formula": null,
  "task": "arithmetic"
}
```

这不是让 teacher 代替 verifier 猜答案，而是让它围绕可靠答案生成短解释。这样
可以研究：

```text
同一个正确 final answer
直接模板 vs 更自然的 teacher explanation
```

而不把 API 偶然算错混入 gold label。

## 6. sequence answer 的接受门

provider final response 必须是恰好三个字段的 JSON：

```json
{
  "answer_text": "Multiply 12 by 7. The answer is 84.",
  "final_answer": "84",
  "unit": null
}
```

只有同时满足以下条件才转为 SFT：

- `final_answer` 与本地 exact answer 数值相等；
- physics unit 与 canonical unit 完全一致；
- final number 和 unit 实际出现在 `answer_text`；
- 不含 `<USER>`、`<ASSISTANT>` 或其他 special-token 字样；
- 纯英文 ASCII；
- 完整 `<BOS><USER>...<ASSISTANT>...<EOS>` 不超过 256 tokens；
- family 属于固定 training bucket；
- family 不出现在 Lesson 12 冻结 evaluation 文件；
- record/family 不重复。

外部模型的 private `reasoning_content` 不进入训练记录。我们只保留：

- verified final sequence；
- provider model 与 request ID；
- prompt/completion token counts；
- 本地 verifier 的接受或拒绝原因。

## 7. 为什么 sequence 只增强 train split

如果新 teacher 数据同时替换 validation/test，就无法判断 loss 改善来自更好的
训练，还是更容易的新评测。另一方面，若只用 512 条 teacher sequence 完全替换
原先约 1.86 万条 SFT train，固定 409.6 万 sampled tokens 会把小语料反复采样
几十个 epoch，结果混入严重过拟合。因此正式构建器做精确拼接：

```text
train       = reference SFT train + verified external sequence
validation  = byte-identical Lesson 12 SFT validation
test        = byte-identical Lesson 12 SFT test
```

manifest 同时保存组件 hash。新增 teacher sequence 是训练变量，原 hard-label
监督不被删除；checkpoint selection 与最终 SFT loss 仍在原来的 reference split
上完成。只用 512 条 sequence 的纯小语料路线可作为独立 overfit diagnostic，
不混入四条主比较。

## 8. 为什么从 512 pilot 扩到 4,096 条

最初的 512 条只占原 SFT 约 18,600 条记录的 2.7%，主要能验证 API、verifier
和数据拼接管线，作为能力干预偏弱。正式主实验扩到：

```text
arithmetic       2,048
physics_numeric  2,048
total            4,096
```

它约等于原 SFT 记录数的 22%，同时仍保留全部 reference hard labels。若这一级
规模依然没有提高冻结 exact accuracy，下一步应先分析接受率、family 覆盖和错误
模式，而不是未经诊断继续成倍购买相似文本。

真实付费调用前，先生成两份完全独立的请求计划：

| 项目 | 观察值 |
|---|---:|
| arithmetic families | 2,048 |
| physics families | 2,048 |
| total requests | 4,096 |
| request-plan bytes | 4,058,710 |
| network calls | 0 |
| 两次 plan byte-identical | 是 |

请求计划 SHA-256：

```text
081f151b0c04ea58727a2a9409067cb9b61d147cd0e38c76988ca920fa870d01
```

root manifest SHA-256：

```text
51abc7320c840497e7c012f8d3b9a7ff17f869c850bd73e4f881fc4ec5b7c62e
```

冻结上限为 4,096 次请求、每次最多 512 input / 1024 output tokens，按执行日
V4-Pro cache-miss 价格计算的保守上界不超过约 `$4.80`。实际账单必须以后端
usage 为准，不能把这个上界当成已消费金额。

正式调用使用 16 个并发 worker，并为每个 family 原子写入一份经过解析且
secret-free 的公开答案缓存。缓存不含 authorization、原始 provider body 或
`reasoning_content`。中断后重跑只请求缺失 family；最终数据仍按固定 problem
顺序组装，所以并发完成顺序不会改变训练 corpus。

## 9. local teacher 的规格

Local teacher 仍使用 `DirectSmallGPT` 的同一数值定义，只增大容量：

| 项目 | Student | Local teacher |
|---|---:|---:|
| layers | 6 | 12 |
| heads | 6 | 10 |
| width | 384 | 640 |
| context | 256 | 256 |
| vocabulary | 264 | 264 |
| parameters | 10,821,504 | 59,331,200 |
| raw FP32 parameter bytes | 43,286,016 | 237,324,800 |

它不准备部署到 Nspire，因此不受 4–6 MiB 文件门约束。它的任务是在 Host GPU
上提供更丰富的 soft targets；最终仍只部署 student。

为了让 teacher 的训练量可审计：

```text
Local-Teacher-CPT
  2,000 updates * 4,096 tokens = 8,192,000 tokens

Local-Teacher-SFT
  1,000 updates * 4,096 tokens = 4,096,000 tokens
```

pretrain 使用 Lesson 12 的 replay-aware CPT mixture；SFT 使用完全相同的
role-aware assistant mask。teacher checkpoint 同样保存父 hash、route、
tokenizer、模型配置和 source commit。

## 10. hard CE 与 soft KL

设 student logits 为 \(z_s\)，local teacher logits 为 \(z_t\)，温度为 \(T\)：

```text
p_t = softmax(z_t / T)
log p_s = log_softmax(z_s / T)

soft_loss = T^2 * KL(p_t || p_s)
hard_loss = CE(z_s, gold)

total = (1 - alpha) * hard_loss + alpha * soft_loss
```

本课固定：

```text
T = 2.0
alpha = 0.5
```

`T > 1` 会把分布变软。例如 gold 是字符 `8` 时，teacher 也许认为 `7`、`9`、
`.` 比字母 `q` 更合理；hard label 只告诉 student “选 8”，soft distribution
还能表达这种相对结构。乘 \(T^2\) 用于补偿温度导致的梯度缩小。

## 11. assistant mask 也必须作用于 KL

Lesson 12 已经规定 USER question 不计 hard CE。本课如果只 mask CE，却让 KL
覆盖 USER bytes，就会出现隐藏的不公平目标：

```text
hard CE: assistant only
soft KL: entire question + answer   [错误]
```

正确实现：

```text
mask = 1 only on assistant answer bytes and EOS

hard = sum(mask * CE_per_token) / sum(mask)
soft = sum(mask * KL_per_token) / sum(mask)
```

测试会把所有 USER position 的 student/teacher logits 改成极端相反值，确认：

- hard loss 不变；
- soft loss 不变；
- USER logits gradient 精确为 0；
- assistant logits gradient 非 0；
- teacher 参数永久 `requires_grad = false` 且没有 gradient。

这才是“唯一主要变量是 soft target”的同架构比较。

## 12. 训练结果

### 12.1 Local teacher

RTX 5080 Laptop GPU 上的结果：

| Stage | updates | sampled tokens | full val | full test | wall time |
|---|---:|---:|---:|---:|---:|
| Local-Teacher-CPT | 2,000 | 8,192,000 | 1.3798 | 1.2746 | 625.0 s |
| Local-Teacher-SFT | 1,000 | 4,096,000 | 0.2816 | 0.2827 | 280.2 s |

两个 stage 的最佳点都在最后一步。CPT optimizer throughput 约
`13,171 token/s`，SFT 约 `14,766 token/s`，两者峰值 CUDA allocation 都约
`1.37 GB`。SFT checkpoint：

```text
0184c1fde793c0677db8c41c9a49ba5f705a45090525147e48926a609167198b
```

严格 reload 验证了 76 个 state tensor、59,331,200 参数、finite values、
264-token contract 和 tied embedding identity。

### 12.2 更低 loss，却没有更高自由生成准确率

Local teacher 的 SFT full validation loss 明显低于 ordinary student：

```text
ordinary 10.8M SFT  0.4683
local 59.3M teacher 0.2816
```

但同一冻结 128-prompt greedy gate：

| Model | format valid | exact task accuracy |
|---|---:|---:|
| Ordinary SFT student | 95.31% | 2/128 |
| Local SFT teacher | 95.31% | 1/128 |

teacher 分任务：

| Task | exact |
|---|---:|
| mixed arithmetic | 0/32 |
| easy arithmetic | 1/32 |
| GSM8K | 0/32 |
| physics numeric | 0/32 |

它唯一答对的是：

```text
Calculate 8 + 2.
-> The answer is 10.
```

这再次说明 teacher-forced loss 与 free generation 是不同测量。训练 loss 问：

```text
给定正确 answer prefix，下一个 byte 的概率高不高？
```

greedy exact gate 问：

```text
没有 gold answer prefix 时，整条生成轨迹是否最终落在正确数值？
```

一个早期错误 byte 会改变后续全部条件分布。更大模型可以更好地拟合答案语言与
局部数字结构，却仍没有学会可靠算法。它依然可以作为 soft-logit teacher，但
我们不再预设其 KL 一定提高 exact accuracy。

### 12.3 Student distillation

`Local-Logit-Distilled-SFT` 已按冻结合同完成：

| 项目 | Ordinary SFT | Local-logit distilled |
|---|---:|---:|
| parameters | 10,821,504 | 10,821,504 |
| parent | 同一个 CPT | 同一个 CPT |
| optimizer updates | 1,000 | 1,000 |
| sampled tokens | 4,096,000 | 4,096,000 |
| full validation loss | 0.4683 | **0.4448** |
| full test loss | 0.4632 | **0.4441** |
| frozen exact | **2/128** | 1/128 |
| format valid | 95.31% | 95.31% |

蒸馏 student 的 sampled validation loss 从 `1.3664` 降到 `0.3771`，最佳点在
第 1,000 步。训练最后一次记录为：

```text
hard CE  0.46849
soft KL  0.55475
total    0.51162
```

checkpoint SHA-256：

```text
5b411ff34b70869a0b609346b4b11464890d34d3b9b71af4ff00f2873a90a2d1
```

文件 hash 与 manifest 一致，严格重载验证了固定 `6/6/384`、10,821,504
参数和 40 个 state tensor。训练 wall time 为 `128.4 s`，student update
throughput 约 `32,265 token/s`；单独测得 local teacher forward throughput
约 `82,775 token/s`。

冻结评测唯一答对：

```text
Calculate 6 * 2.
-> The answer is 12.
```

分任务仍是 mixed arithmetic `0/32`、easy arithmetic `1/32`、GSM8K `0/32`、
physics `0/32`。这是一项明确的负结果：

```text
soft target 改善了 teacher-forced token distribution
!=
student 学会了可泛化的计算过程
```

### 12.4 输出吸引子变了，但算法能力没有出现

三个模型在同一 128 题上的最常见 completion 都是：

```text
The answer is 10.
```

但集中程度不同：

| Model | 该 completion 次数 | 占比 | unique completions |
|---|---:|---:|---:|
| Ordinary SFT student | 33 | 25.78% | 73 |
| Local SFT teacher | 47 | 36.72% | 61 |
| Local-logit student | 28 | 21.88% | 74 |

因此不能把 Logit student 的失败简单解释成“逐字复制 Teacher 的 `10`
吸引子”。`alpha=0.5` 的 hard CE、soft KL、有限 student 容量和优化轨迹共同
产生了一个不同分布：它比 Teacher 更分散，却依然不会可靠计算。这也说明：

1. 输出多样性不等于答案正确；
2. teacher 的低 loss 不保证 teacher 本身有可蒸馏的算法能力；
3. KL 可以传递局部 token 相似性，却不会凭空创造训练数据中没有稳定体现的
   中间计算机制；
4. 下一步的 external synthetic-data generator 必须提供经过验证、覆盖更多
   family 的正确解释，而不能只换一个更强的 loss。

外部 sequence 与 combined 路线必须等 runtime credential 存在并且 512 条输出
通过 verifier 后才能执行；未运行的 route 不以设计值冒充观察值。

## 13. 当前能与不能声称什么

代码和 dry-run 已经能够证明：

- provider/current model contract 已更新；
- secret 不进入提交物或可序列化配置；
- external family 不泄漏到冻结 evaluation；
- 外部 sequence 会经过 exact value/unit verification；
- local teacher 与 student 逐 token vocab 对齐；
- hard/soft loss 使用同一个 assistant mask；
- 三个新 student route 共享架构、parent 与 token budget；
- local-logit route 的 loss 改善没有转化成 exact-answer 改善。

在正式结果完成前仍不能声称：

- V4-Pro sequence 已经生成或花费了多少；
- verified synthetic-data SFT 提高了 exact accuracy；
- local teacher 具备可泛化的可靠计算能力；
- KL 提高冻结 exact accuracy；
- combined route 一定叠加收益；
- 任何新模型已经量化、进入 Host C 或 Nspire。

下一课才在这些可归因结果上比较 direct answer 与受限 CoT 的 SFT/RLVR；不能用
RL 阶段补写本课没有得到的 teacher 证据。
