# nanoGPT for Nspire

这是一个边学 Transformer、边把小型 GPT 推理移植到 TI-Nspire CX II CAS 的项目。

当前路线从 Tiny Shakespeare 字符模型开始，依次比较直接训练小模型、量化模型和蒸馏小模型；通过 Host C 数值对齐后，再进入 Ndless 真机推理。官方 nanoGPT 源码保存在 [`upstream/nanoGPT/`](upstream/nanoGPT/)，不会与我们的实现混写。

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
- 已完成（Host/ARM）：[第九课：Ndless 像素对话界面、隐私生命周期与真机部署](docs/lessons/09-ndless-pixel-chat-ui.md)
- 待真机门：上传读回、启动、CX II tokens/s/峰值 RAM 与退出显示恢复

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
完整 `nanogpt-chat.tns`。Host 现有 7 个 CTest；ARM chat ELF 为 `94,048 bytes`，
封装后的 `.tns` 为 `59,310 bytes`。默认部署 bundle 使用 Quantized-Small：

```text
dist/
├── nanogpt-chat.tns    59,310 bytes
└── model.ngm.tns    6,036,544 bytes
```

真实 CX II 已被检测并成功列过根目录，但第一次 6 MiB 原子同步因计算器息屏触发
LibUSB error，失败后的远端目录尚未重新审计。因此当前只声明 Host/ARM 通过，
不把真机上传、启动、速度或峰值内存写成已完成。

三条可部署小模型路线与两层公平性已经冻结在
[`small-model-comparison-design.md`](docs/plans/2026-07-27-small-model-comparison-design.md)，
机器可读状态表位于
[`experiments/small-model-comparison.json`](experiments/small-model-comparison.json)。

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
