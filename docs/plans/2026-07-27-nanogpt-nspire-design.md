# nanoGPT for Nspire 设计

日期：2026-07-27  
状态：已由用户确认

## 1. 项目目标

本项目用一条可以逐步验证的学习路线，把 nanoGPT 从 PyTorch 训练程序变成能在 TI-Nspire CX II CAS 上生成文本的原生推理程序。

第一阶段采用 Tiny Shakespeare 和字符级 tokenizer。它的目的不是立即做出物理问答助手，而是完整走通：

1. 数据准备与 tokenizer；
2. Transformer 训练与采样；
3. 直接训练小模型、量化和知识蒸馏；
4. PyTorch 模型导出；
5. 纯 C Host 推理与数值对齐；
6. Ndless 交叉编译与真机测量。

Phy-nspire 英文文档、监督后训练和 RL 属于第二阶段。只有基础推理、物理语料版本管理、SFT 以及可自动检查的物理奖励函数都成立后，才能进入 RL。语言流畅不能当作物理正确。

以后还会从零重写一遍 Transformer 算子，作为单独的深入学习路线；它不阻塞当前的端到端路线。

## 2. 已知设备边界

目标设备是 TI-Nspire CX II CAS，运行 OS 6.4.0.74 和 Ndless r2022。相邻的 NspirePhysics 项目已经测得：

- 总 RAM：35,650,680 bytes，测量时空闲 31,805,820 bytes（30.33 MiB）；
- 总 Flash：96,862,208 bytes，测量时空闲 91,090,944 bytes（86.87 MiB）；
- 设备程序使用 Ndless SDK 和原生 ARM C/C++；
- Windows 上的 Host 工作可以直接运行，Ndless 设备构建使用 WSL。

首轮比较采用以下保守预算：

- 可部署模型权重文件：4–6 MiB；
- 推理峰值 RAM：不超过 24 MiB；
- 启动前必须计算完整内存需求；
- 生成循环使用有上限的预分配 arena，不依赖无界动态扩容。

这里的“小模型”首先表示可以在这个文件和内存预算内独立部署，而不只表示参数数量少。

## 3. 共同实验基线

所有实验固定使用：

- Tiny Shakespeare 原始文本；
- 按字符排序得到的 65 字符词表；
- 90%/10% 的顺序训练/验证切分；
- 相同的上下文长度、训练 token 预算、随机种子和采样提示；
- 带 SHA-256 的原始数据和配置清单；
- 验证集 loss 和 bits-per-character 作为主要质量指标。

官方 `upstream/nanoGPT/` 保持只读参考。我们的训练、导出和推理代码放在项目自己的目录中，不直接混入官方源码。

## 4. 三条小模型路线

### 4.1 Direct-Small

选择能在设备预算内运行的 student 架构，从随机参数直接训练。它给出“少量高精度参数”的基础结果。

### 4.2 Quantized-Small

先在电脑上训练更大的 teacher，再把权重转换成 INT8 或 INT4。设备 runtime 必须直接使用整数表示和受控的缩放参数，不能在启动后把全部权重展开成 FP32。

这条路线回答：在相同存储和内存预算下，更多低精度参数是否优于更少高精度参数。

### 4.3 Distilled-Small

student 的架构和数值精度与 Direct-Small 完全相同。训练时同时学习真实 next-token 标签和 teacher 的 soft logits。

Direct-Small 与 Distilled-Small 的唯一主要变量是训练目标，因此二者可以隔离出知识蒸馏本身的收益。

### 4.4 组合实验

完成三条基础路线后，再做 Distilled-Small 加量化。它用于寻找最佳部署模型，但不混入基础三路线结论，以免无法判断收益来源。

## 5. 公平比较协议

比较同时使用两种公平性：

1. **同架构比较**：Direct-Small 对 Distilled-Small，用于测量蒸馏收益；
2. **同部署预算比较**：Direct-Small、Quantized-Small 和 Distilled-Small 都满足同一文件与峰值 RAM 上限，用于选择真机方案。

每次实验必须记录：

- 模型结构和参数量；
- 原始权重与部署权重的字节数；
- 训练配置、随机种子、数据哈希和代码提交；
- validation loss 与 bits-per-character；
- 固定提示、固定 seed 的生成文本；
- 训练时间、导出时间和 Host 推理速度；
- `.tns`、模型文件、首字符延迟、字符/秒和真机峰值 tracked heap；
- PyTorch 与 C logits 的最大绝对/相对误差；
- greedy 解码的 token 序列是否一致。

生成文本只作定性辅助，不代替验证集指标和真机测量。

## 6. 仓库结构

```text
nanoGPT_for_Nspire/
├── upstream/nanoGPT/       官方源码，只作参考和对照
├── training/               PyTorch 训练、量化、蒸馏
│   ├── configs/
│   └── nanogpt_nspire/
├── runtime/                不依赖 PyTorch 的 C 推理核心
│   ├── include/
│   ├── src/
│   └── platform/
│       ├── host/
│       └── ndless/
├── tools/                  模型导出、检查和大小分析
├── tests/                  Python/C 数值一致性与回归测试
├── experiments/            可提交的配置与结果摘要
├── docs/
│   ├── lessons/            中文 Transformer/NLP 课程
│   └── plans/              设计与实施计划
└── artifacts/              数据、checkpoint、模型和构建产物，不进 Git
```

`artifacts/` 可以在本地生成，但必须由根目录 `.gitignore` 排除。值得长期复核的实验结果以小型 JSON/Markdown 摘要进入 `experiments/`，大 checkpoint 不进入 Git。

## 7. 数据流和模型文件

主数据流为：

```text
Tiny Shakespeare
  -> 字符 tokenizer
  -> PyTorch 训练
  -> checkpoint
  -> 量化或蒸馏
  -> 统一模型文件
  -> Host C 推理
  -> 数值对齐
  -> Ndless 编译
  -> 真机生成与测量
```

统一模型文件至少包含：

- 格式版本和端序；
- 模型维度和上下文上限；
- 有序 token 表；
- 数值精度与量化方案；
- 每个张量的类型、形状、偏移和长度；
- 文件长度与完整性校验值。

加载器必须先验证整个头部和所需内存，再让任何张量参与计算。

## 8. 学习顺序

每个可运行阶段配一篇中文课程，解释函数、张量形状、数据流、设计理由以及它与最终真机目标的关系。

1. 字符、token、词表和训练样本；
2. embedding、logits 与交叉熵；
3. causal self-attention 和张量形状；
4. 完整训练循环与过拟合；
5. 量化；
6. 蒸馏；
7. C 推理与 PyTorch 对齐；
8. CX II 内存与性能测量。

课程代码优先保持小、可读和可测试。优化版实现必须保留一个容易理解的参考路径，除非设备约束明确要求删除。

## 9. 验收门与失败处理

### 9.1 PyTorch 基线

- 数据源哈希固定；
- 固定 seed 可以复现实验；
- loss 明显下降；
- checkpoint 可以重新加载和采样。

### 9.2 三模型实验

- 配置、参数量、日志和指标齐全；
- 超出部署文件或峰值 RAM 预算的模型标记为不可部署；
- 不因生成样例看起来较好而忽略定量失败。

### 9.3 模型导出

以下情况必须拒绝加载：

- 未知格式版本或量化类型；
- 文件截断、偏移越界或长度溢出；
- 张量维度不匹配；
- 完整性校验失败；
- 计算出的内存需求超过配置上限。

### 9.4 Host C 对齐

- FP32 logits 在预先规定的误差内对齐 PyTorch；
- 整数量化模型明确记录量化误差；
- 固定 greedy 解码产生相同 token 序列；
- malformed model 测试不能崩溃或越界。

### 9.5 Nspire 真机

- 模型过大、上下文过长或内存不足时显示明确错误并安全返回；
- Host、模拟器或链接成功不能替代真机运行；
- 真机验收必须记录文件大小、加载时间、速度和 tracked heap；
- 程序退出后应正常返回 Documents，不以黑屏或重启作为可接受失败。

## 10. 当前不做

首阶段不做以下内容：

- 中文 tokenizer 和完整 CJK 字体；
- Phy-nspire 文档后训练；
- RL 或偏好优化；
- 通用聊天模板和多轮对话；
- GPU/CUDA 优化；
- 在 Nspire 上训练；
- 未经 Host 数值对齐就直接调真机结果。

这些内容只有在前置验收门关闭后才进入新的设计和实施计划。
