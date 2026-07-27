# nanoGPT for Nspire

这是一个边学 Transformer、边把小型 GPT 推理移植到 TI-Nspire CX II CAS 的项目。

当前路线从 Tiny Shakespeare 字符模型开始，依次比较直接训练小模型、量化模型和蒸馏小模型；通过 Host C 数值对齐后，再进入 Ndless 真机推理。官方 nanoGPT 源码保存在 [`upstream/nanoGPT/`](upstream/nanoGPT/)，不会与我们的实现混写。

## 当前进度

- 已确认：[总体设计](docs/plans/2026-07-27-nanogpt-nspire-design.md)
- 已完成：[第一课：字符、token 与数据集](docs/lessons/01-tokenization-and-dataset.md)
- 正在实施：[Lesson 02 embedding 基线计划](docs/plans/2026-07-27-lesson-02-embedding-baseline.md)
- 第二课：[batch、embedding、logits 与 loss](docs/lessons/02-batches-embeddings-and-loss.md)

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
