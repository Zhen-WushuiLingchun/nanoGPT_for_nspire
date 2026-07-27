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
- 后续：Lesson 08 C 对齐、Lesson 09 CX II 测量

Lesson 06 的 Teacher v1 虽优于 Direct-Small，但未通过预注册质量门；INT4
体积门和量化误差门均通过，因此该产物保留为 diagnostic。Lesson 07 只把
dropout 从 `0.2` 提高到 `0.3`：Teacher v2 通过原质量门，正式 Quantized-Small
的存储与质量门也通过。

基础 5000-step Distilled-Small 与 Direct-Small 使用相同架构、初始参数和 student
训练 token，但 validation loss 为 `1.522163`，未胜过 Direct 的 `1.499790`。
独立的 10000-step 扩展实验改善到 `1.506599`，仍未反超且不计入同预算基础比较。
负结果与扩展结果均保留，不做事后改门槛。

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
```

生成的数据、checkpoint、导出模型和构建产物统一放在 `artifacts/`，不会进入 Git。

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
