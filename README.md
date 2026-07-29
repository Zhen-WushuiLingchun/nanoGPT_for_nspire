# nanoGPT for Nspire

> **项目状态：已完成（Research Complete，2026-07-29）**
>
> 本仓库作为“从 Transformer 基础到 TI-Nspire 真机部署”的教学与研究原型
> 已结项。最终模型不是可靠的通用助手；失败模式、负结果和真机边界均作为项目
> 结论保留。

这是一个边学 Transformer、边把小型 GPT 推理移植到 TI-Nspire CX II CAS 的项目。

第一阶段从 Tiny Shakespeare 字符模型开始，依次比较直接训练小模型、量化模型和
蒸馏小模型；通过 Host C 数值对齐后，再进入 Ndless 真机推理。第二阶段已完成
English-only 数学物理助手：byte tokenizer、英语 Base、角色 SFT、强 teacher
蒸馏、可验证 RL/CoT、重新量化与真机部署。官方 nanoGPT 源码保存在
[`upstream/nanoGPT/`](upstream/nanoGPT/)，不会与我们的实现混写。

## 当前进度

- 已确认：[总体设计](docs/plans/2026-07-27-nanogpt-nspire-design.md)
- 已完成：[第一课：字符、token 与数据集](docs/lessons/01-tokenization-and-dataset.md)
- 已完成：[第二课：batch、embedding、logits 与 loss](docs/lessons/02-batches-embeddings-and-loss.md)
- 已完成：[第三课：单头 causal self-attention](docs/lessons/03-causal-self-attention.md)
- 已完成：[第四课：完整训练循环与过拟合](docs/lessons/04-training-loop-and-overfitting.md)
- 已完成：[第五课：Direct-Small 完整小 GPT](docs/lessons/05-direct-small-gpt.md)
- 已完成：[第六课：Teacher 与 INT4 量化诊断](docs/lessons/06-int4-quantization.md)
- 已完成：[第七课：Teacher v2、正式量化与蒸馏](docs/lessons/07-teacher-v2-distillation-and-comparison.md)
- 已完成：[第八课：统一模型文件、C 推理与 PyTorch 对齐](docs/lessons/08-c-runtime-and-pytorch-alignment.md)
- 已完成（Host/ARM/真机启动）：[第九课：Ndless 像素对话界面、隐私生命周期与真机部署](docs/lessons/09-ndless-pixel-chat-ui.md)
- 已完成：[第十课：English byte tokenizer、角色 token 与可审计语料地基](docs/lessons/10-english-byte-tokenizer-and-corpus.md)
- 已完成：[第十一课：CUDA、模型预算、公开语料与首个 English Base pilot](docs/lessons/11-english-base-pilot.md)
- 已完成：[第十二课：数学物理 CPT、角色 SFT 与精确能力诊断](docs/lessons/12-math-physics-cpt-and-sft.md)
- 已完成：[第十三课：外部合成数据 SFT、本地 logit teacher 与严格蒸馏](docs/lessons/13-external-and-local-teachers.md)
- 已完成：[第十四课：可控短 CoT、固定 token 公平比较与 512 context](docs/lessons/14-controllable-cot-and-context.md)
- 已完成：[第十五课：256-token 推理、GQA、ALiBi 与计算器边界](docs/lessons/15-long-output-gqa-alibi.md)
- 已完成：[第十六课：紧凑可验证 SFT v2、终止边界与“自洽但算错”](docs/lessons/16-compact-verified-sft-v2.md)
- 已完成：[第十七课：RLVR、direct-RLAIF 与奖励稀疏/奖励错位](docs/lessons/17-rlvr-rlaif-and-reward-hacking.md)
- 已完成（Host/ARM/传输/真机启动）：[第十八课：GQA-ALiBi W4A8、可控 CoT 界面与真机部署](docs/lessons/18-gqa-alibi-cot-device-pilot.md)
- 已结项：18 课研究路线、量化 C runtime、Ndless UI 和两代真机部署均已完成

## 结项结论

这个项目的目标不是制造一个可以与现代云端 LLM 竞争的计算器助手，而是完整走通
并测量一条受极端存储、RAM 和算力限制的语言模型路线。结项证据包括：

- 从字符模型到 byte tokenizer、Base/CPT/SFT、teacher-logit 蒸馏、可控 CoT、
  RLVR/direct-RLAIF 的 18 课实验链；
- Direct-Small、Quantized-Small、Distilled-Small 的同架构与同部署预算比较；
- packed W4A8、不展开 FP32 matrix 的 Host C 推理和 PyTorch/C 对齐；
- Ndless 像素对话界面、volatile New Chat/Exit、TTFT/tokens/s/context/tracked
  RAM 显示和独立模型文件；
- 旧 Quantized-Small 真机照片的三组观察值：约 `0.9–1.2 char-token/s`、
  `5–15 s TTFT`、`8.2 MiB tracked RAM`；
- prompt-ending 修复版在真机对不同 prompt 产生不同 continuation，并逐字符
  匹配 Host/Python reference；
- Lesson 16 GQA-ALiBi SFT v2 已量化为 5,387,968-byte `.ngm v2`，与新版
  62,310-byte Ndless 程序一起通过真机完整 SHA-256 回读；用户随后确认新版能
  启动并完成 Direct 生成；
- 设备端最终只保留两个正式文件，没有 `.upload` 或 `.previous` 副本。

三张照片的速度与 TTFT 来自不同 prompt/context，因此是重复真机观察范围，
不是固定 prompt 的严格 benchmark。`8.2 MiB` 是程序已知内存区域的 tracked
总量，不是计算器 OS 级 peak heap；退出后的 Documents/LCD 恢复也没有单独
拍照留档。它们作为测量边界保留，但不再阻塞教学研究项目结项。

最终能力结论同样属于成果：约 9.5M 参数的 byte-level 模型能学会角色、格式、
`<THINK>/<FINAL>` 控制和局部英文，却没有学会可靠的开放域语言或可泛化数学
推理。SFT、严格 logit 蒸馏、合成数据、RLVR 和 RLAIF 都没有消除这个容量/
数据/优化边界；看似流畅的 CoT 也不等于正确推理。

Lesson 06 的 Teacher v1 虽优于 Direct-Small，但未通过预注册质量门；INT4
体积门和量化误差门均通过，因此该产物保留为 diagnostic。Lesson 07 只把
dropout 从 `0.2` 提高到 `0.3`：Teacher v2 通过原质量门，正式 Quantized-Small
的存储与质量门也通过。

基础 5000-step Distilled-Small 与 Direct-Small 使用相同架构、初始参数和 student
训练 token，但 validation loss 为 `1.522163`，未胜过 Direct 的 `1.499790`。
独立的 10000-step 扩展实验改善到 `1.506599`，仍未反超且不计入同预算基础比较。
负结果与扩展结果均保留，不做事后改门槛。

Lesson 08 已把三条路线导出成同一 `.ngm` 格式。Direct/Distilled FP32 和
Quantized-Small packed W4A8 均通过 Host C logits 与 64-token 贪心对齐；
Quantized-Small 不展开 FP32 matrix，模型加 runtime arena 为 `8,415,168 bytes`，
固定 validation loss 为 `1.473743`。完整 runtime 也已通过 Ndless ARM
compile/link/package，生成 `48,056-byte` smoke `.tns`。这些仍不是 CX II 真机
速度或峰值内存。

Lesson 09 已加入固定容量多轮 cell、逐 token prefill/decode、320×240 RGB565
像素界面、TTFT/tokens/s/context/tracked RAM、New Chat/Exit volatile zeroing 和
完整 `nanogpt-chat.tns`。Host 现有 7 个 CTest；prompt-ending 修复版 ARM chat
ELF 为 `94,064 bytes`，封装后的 `.tns` 为 `59,309 bytes`。默认部署 bundle
使用 Quantized-Small：

```text
dist/
├── nanogpt-chat.tns    59,309 bytes
└── model.ngm.tns    6,036,544 bytes
```

真实 CX II 已完成两个文件的单进程原子同步：上传后完整读回的 SHA-256 与本地
一致，设备端最终只保留正式文件。该次同步耗时 `135.2 s`。此前失败不是息屏或
必须物理插拔，而是 USB/IP 尚未完成 WSL 枚举，以及连续启动独立 `phy-nlinkctl`
进程造成的 TI 文件服务握手竞态。原版应用也已经在真机启动并打开 W4A8 模型；
照片证据显示约 `0.9–1.2 char-token/s`、`12–15 s TTFT` 和 `8.2 MiB tracked
RAM`。统一尾随换行造成的固定句首已修复；修复版真机的 `hello` 与
`one plus two` 输出逐字符匹配 Host/Python reference，但仍频繁进入 `the state`
吸引盆。49-prompt 扫描中 Direct/Teacher/Quantized/Distilled 的 `state` 命中数
分别为 `4/15/13/46`；当前 W4A8 的 temperature/top-k stress probe 则只有
`1/160`。完整原因、蒸馏负结果和解码边界记录在 Lesson 09。

三条可部署小模型路线与两层公平性已经冻结在
[`small-model-comparison-design.md`](docs/plans/2026-07-27-small-model-comparison-design.md)，
机器可读状态表位于
[`experiments/small-model-comparison.json`](experiments/small-model-comparison.json)。

Lesson 10 冻结了 `256 byte + 8 special token` 的 264-token 合同，使
`USER/ASSISTANT` 第一次成为模型可见 token，并加入 assistant-only SFT loss
mask、公开数据许可门、family-level 防泄漏 split 和不使用 `eval` 的精确算术
生成器。256 个 arithmetic family / 512 个 record 的两次 smoke build 全文件
byte-identical；在 Lesson 10 结束时这仍只是数据地基，当时尚未训练英语 Base。
长期设计见
[`english-math-physics-assistant-design.md`](docs/plans/2026-07-28-english-math-physics-assistant-design.md)。

Lesson 11 已建立独立 CUDA 12.8/PyTorch 2.11 环境，冻结
`6 layers / 6 heads / width 384 / context 256 / vocab 264` 的
10,821,504-parameter Student，以及电脑端 `12×640` Teacher。两个精确
Parquet snapshot 构成 848-document / 4,312,602-token public pilot；独立重建
manifest SHA-256 一致。首个真实英语 Base 在 RTX 5080 Laptop 上训练
4,096,000 tokens，完整 validation loss 从 `5.733047` 降到 `2.119973`，
完整 test loss 为 `2.064447`，约 `90,281 tokens/s`，峰值 CUDA allocation
`393,493,504 bytes`。生成已经出现英文局部结构但仍是无可靠语义的拟词，因此
它是 Base LM，**还不是聊天、数学或物理助手**。W4 文件静态估算
`6,172,992 bytes`，距离 6 MiB 门仅 `118,464 bytes`；真实 `.ngm v2`、C 对齐
和 Nspire 部署仍待后续课实测。

Lesson 12 固定了 GSM8K/OASST1 与 12,000 个算术、4,000 个物理 family，
构建 `6,718,812-token` replay-aware CPT 和 `2,793,452-token` role-aware
SFT。CPT 把 domain validation loss 从 `2.4197` 降到 `0.7892`；SFT 把
assistant-only loss 降到 `0.4683`。在 Base/CPT/SFT 共用的 128-prompt
greedy gate 上，格式有效率从 `0%`、`47.7%` 提高到 `95.3%`，但 SFT 只精确
答对 `2/128`，easy arithmetic 也只有 `1/32`。因此它已经是模型可见 role
token 驱动的窄 instruction baseline，却仍不是可靠数学物理助手。这个
“会回答格式、不会泛化计算”的负结果将作为 Lesson 13 teacher/distillation
的固定起点，不用低 loss 掩盖。

Lesson 13 已完成共享 tokenizer 的 59.3M Local Teacher 和同架构
Local-Logit-Distilled student。Teacher 的 full validation loss 为 `0.2816`，
但冻结精确题仅 `1/128`；Logit 蒸馏把 10.8M student 的 loss 从 `0.4683`
改善到 `0.4448`，精确题仍为 `1/128`。这说明 soft target 确实改变了 token
分布，却没有凭空产生可泛化算术算法。外部 V4-Pro 路线明确属于 verified
synthetic-data SFT，而非严格蒸馏；4,096 次正式调用得到 4,077 条通过 exact
value/unit/context 门的序列，成功响应 token 的价格估算约 `$1.70`，且
validation/test 与 reference 字节一致。Synthetic-SFT 把格式有效率提高到
97.66%，但 exact 仍为 `2/128`；合成数据再叠加 local-logit 后为 `1/128`。
因此更丰富的正确解释和 soft likelihood 都改变了输出分布，却仍未让当前
10.8M byte-level student 学会可泛化的计算算法。

Lesson 14 使用已有 `<THINK>` / `<FINAL>` token 训练 Direct、Short-CoT 与
Hybrid-Control 三条同架构、同 4,096,000 sampled-token 路线。48-token 冻结
评测中，三者分别为 `1/128`、`0/128`，Hybrid 用 direct/think cue 都是
`2/128`；96-token 诊断显著减少 CoT 截断，但没有增加正确题。因此本课证明
SFT 能教会可切换的输出格式，却没有证明 10.8M byte model 学会了可泛化推理。

独立的 256→512 context pilot 保留旧位置、复制初始化新位置并做 250-step
长序列 CPT。新位置 validation loss 从 `2.8666` 降到 `1.5645`，旧位置从
`1.5312` 小幅变为 `1.5412`。当前 FP32 MHA KV cache 也从 `4.50 MiB` 翻倍到
`9.00 MiB`；该 checkpoint 仅在 PyTorch 端验证，现有 C loader 仍拒绝
`block_size > 128`，所以尚未宣称完成 Nspire 导出、数值对齐或真机部署。

Lesson 15 把 Direct 与 Think 的主输出预算都提高到 256 token。在相同 128 题上，
512-MHA 的 Direct/Think 为 `2/128` 与 `1/128`；两条 2-group GQA 路线的
Direct/Think 都为 `2/128`。256 token 消除了 context truncation，却没有让更长
CoT 变成可靠计算。GQA 把 512-context FP32 KV 理论预算从 `9.00 MiB` 降到
`3.00 MiB`；同预算 uptraining/SFT 中，ALiBi 的完整 validation/test loss 为
`0.8196/0.8452`，略优于 MHA 的 `0.8282/0.8531`，但仍只是 PyTorch 研究
checkpoint。PyTorch 原生 GQA kernel 在这套小形状诊断中更慢，且并非 Nspire
执行环境，因此未保留；后续仍需量化 GQA C kernel、Host 对齐和真机测量。

Lesson 16 用 24,000 个算术、12,000 个数值物理 family 和 5,402 条通过本地
逐步精确检查的 GSM8K 注释构建紧凑 SFT v2，并对 `<FINAL>` / `<EOS>` 使用
4 倍加权 CE。Think 的格式、模式、终止和 256-token 截断从
`75.78% / 75.78% / 20.31%` 改善到 `100% / 100% / 0%`，但冻结主评测仍是
`2/128`。新 256-family 挑战集上父模型 Direct/Think 都为 `3/256`，SFT v2
反而只有 `1/256`；即使 v2-Think 有 98.83% 输出复用题目数字、64.06%
reasoning/final 局部一致，也没有学会可靠计算。因此下一阶段保留父模型与 v2
双起点：数值能力主线使用本地 exact/unit verifier 做 RLVR，DeepSeek AI
feedback 独立用于解释清晰度和物理合理性；不把 API 合成答案写成严格蒸馏。

Lesson 17 从无更新 screen 选择了 mixed-exact group 更多的 Lesson 15
checkpoint，随后用三组固定 seed 分别训练 RLVR、direct-RLAIF 和组合路线。
每条路线每 seed 采样 512 个 completion，但 64 个 group 中有 54–57 个奖励
零方差。RLVR、direct-RLAIF、组合路线的 Primary 三种子 mean exact 分别为
`6.00/256`、`3.00/256`、`6.00/256`，Challenge 分别为 `5.33/512`、
`6.67/512`、`5.33/512`；SFT-only 是 `4/256` 与 `6/512`。三条路线都未同时
改善两套，也都未通过 95% format/mode 门，因此不宣称获得可泛化 RL 能力，
不把最好 seed 事后挑出量化或上真机。训练 family 与两份 holdout 的交集均为零。

Lesson 18 没有部署一个失败 gate 后事后挑出的 RL seed，而是把格式最稳定的
Lesson 16 SFT v2 做成 `.ngm v2`。新格式和 C runtime 支持 264-token
byte-special tokenizer、512 context、6:2 GQA 和 ALiBi，5,387,968-byte W4A8
模型加 3,165,312-byte arena 的静态下界为 8.16 MiB。界面用 `Tab` 切换
Direct/Think，并把 `<THINK>` 到 `<FINAL>` 的正文显示为独立 cell。Host 的
Direct 短序列通过严格 logits gate；96-token Think 的贪心序列一致，但累积
logits 误差未通过旧 gate。Host 样例仍会把 `12 times 7` 错答为 `12 N`，而且
FP32 与 W4A8 正文相同，所以这是能力失败，不是量化失败。两个文件已上传到
`/nanoGPT/` 并完整 SHA-256 回读，设备端无 `.previous/.upload`；真机启动、
Direct 生成已由用户报告，但固定 prompt 对齐、Think cell、速度与 RAM 仍待
屏幕实测。额外 Host sweep 证明 prompt 和大小写会改变输出；频繁出现的
`10/100/1000/10000` 是窄 SFT 分布形成的模型吸引盆，而非前端固定忽略输入。

## 快速开始

需要 Python 3.10 或更新版本、PyTorch 2 或更新版本，以及 pytest。

```powershell
python -m pip install -e .
python -m pytest -q
python -m nanogpt_nspire.data fetch `
  --output artifacts/raw/tinyshakespeare.txt
python -m nanogpt_nspire.data prepare `
  --input artifacts/raw/tinyshakespeare.txt `
  --output artifacts/data/tinyshakespeare
python -m nanogpt_nspire.lesson10_data smoke `
  --output artifacts/lesson10-smoke `
  --seed 20260728 `
  --examples 256
python -m nanogpt_nspire.model_budget `
  --output artifacts/lesson11-model-budget.json
python -m nanogpt_nspire.lesson11_data `
  --output artifacts/lesson11-public-pilot `
  --registry experiments/lesson10-public-sources.json `
  --split-seed lesson11-public-v1
python -m nanogpt_nspire.lesson12_data `
  --download-dir artifacts/lesson12-downloads `
  --lesson11-data-dir artifacts/lesson11-public-pilot `
  --output-dir artifacts/lesson12-data `
  --registry-path experiments/lesson10-public-sources.json

cmake -S . -B build/host -G "Visual Studio 17 2022" -A x64
cmake --build build/host --config Release
ctest --test-dir build/host -C Release --output-on-failure
```

生成的数据、checkpoint、导出模型和构建产物统一放在 `artifacts/`，不会进入 Git。

构建 Ndless chat package：

```bash
export NDLESS_SDK="$HOME/.phy-nspire/Ndless/ndless-sdk"
export _NDLESS_TOOLCHAIN_PATH="$HOME/.phy-nspire/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi/bin"
export PATH="$NDLESS_SDK/bin:$_NDLESS_TOOLCHAIN_PATH:/usr/bin:/bin"
make ndless-chat
```

模型作为外部文件放在应用旁边，设备端推荐独立目录：

```text
Documents/nanoGPT/nanogpt-chat.tns
Documents/nanoGPT/model.ngm.tns
```

## 目录职责

- `training/`：PyTorch 训练、量化、蒸馏和数据工具。
- `runtime/`：可移植 C 推理核心，以及 Host/Ndless 平台层。
- `tools/`：模型导出、检查、体积与性能分析。
- `tests/`：Python/C 单元测试与数值一致性测试。
- `experiments/`：可提交的小型实验配置和结果摘要。
- `docs/lessons/`：与代码同步的中文课程。
- `docs/plans/`：经确认的设计和逐阶段实施计划。

## 许可证

本项目自己的代码与文档采用根目录的 [MIT License](LICENSE)。
`upstream/nanoGPT/` 是官方 nanoGPT 源码快照，继续保留其作者 Andrej
Karpathy 与原始 [MIT License](upstream/nanoGPT/LICENSE)；两份版权声明各自适用。
