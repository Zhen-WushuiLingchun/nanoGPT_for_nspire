# Lesson 09 Ndless 像素对话界面与真机测量设计

> 实现状态（2026-07-28）：portable chat state、真实 `.ngm` session、RGB565
> renderer、Host visual fixture、隐私清理测试和 `nanogpt-chat.tns` 均已完成。
> ARM compile/link/package 已通过；真实 CX II 首次同步因设备息屏导致 LibUSB
> error，上传后的远端状态、启动、速度和峰值 RAM 仍待物理重连后验证。实现与证据见
> [`../lessons/09-ndless-pixel-chat-ui.md`](../lessons/09-ndless-pixel-chat-ui.md)
> 和
> [`../../experiments/lesson09-chat-ui.json`](../../experiments/lesson09-chat-ui.json)。

## 目标

在 TI-Nspire CX II CAS 上提供一个原生 Ndless 对话应用：

```text
用户输入 -> 追加 USER cell -> 增量生成 AI cell
         -> 显示实时速度/上下文 -> 用户继续输入
```

退出或 New Chat 时清空全部对话状态，不保存聊天记录。Lesson 09 同时完成真机模型
加载、峰值内存、首 token 延迟和持续生成速度测量。

当前 Tiny Shakespeare checkpoint 是字符续写模型，不是 instruction/chat model。
UI 先建立交互和部署框架；后训练后的物理解释模型可以沿用同一前端。

## 1. 复用隔壁 NspirePhysics 的边界

参考 `../NspirePhysics` 已验证的 Ndless 平台层思路，而不是把其业务 UI 整体复制：

- RGB565 off-screen backbuffer；
- `lcd_blit` 呈现与退出时显示状态恢复；
- keypad/touchpad 采样和小型事件队列；
- touchpad 方向键与 pointer 事件去重；
- 平台资源由单一 init/shutdown 路径拥有；
- Host 可运行的 headless UI/state 测试。

本项目不复用其 notebook/storage 写入层。聊天隐私规则要求不创建 transcript 文件。

## 2. 320×240 像素布局

```text
+--------------------------------------------------+
| nanoCHAT | Q-v2 W4A8 | 37/128                    |  16 px
+--------------------------------------------------+
| [YOU] Explain causal attention.                  |
|                                                  |
| [AI ] Causal attention prevents each position... |
|       ...                                        |  scrollable
|                                                  |
| [YOU] Why use a triangular mask?                 |
| [AI ] ▓                                          |  streaming cursor
+--------------------------------------------------+
| > _                                              |  input
+--------------------------------------------------+
| RUN 12.4 tok/s | TTFT 0.82s | RAM 8.1 MiB        |  15 px
+--------------------------------------------------+
```

视觉规则：

- 1 px 深色边框、无圆角、低色数 pixel aesthetic；
- USER 与 AI 使用不同的低饱和背景和左侧 role tag；
- cell 高度由换行后的文本决定，不使用固定气泡高度；
- transcript 保留逻辑行，屏幕只渲染可见行；
- 正在生成的 AI cell 末尾显示方块 cursor；
- 速度栏固定，不随 transcript 滚动；
- 所有文本和 metrics 都必须在 320×240 下不互相覆盖。

## 3. 状态机

```text
BOOT
  -> LOAD_MODEL
  -> IDLE_EDIT
  -> GENERATING
  -> IDLE_EDIT

任意状态 -> ERROR_DIALOG -> IDLE_EDIT 或 EXITING
IDLE_EDIT/GENERATING -> NEW_CHAT -> IDLE_EDIT
任意状态 -> EXITING -> TERMINATED
```

约束：

- 每次 event-loop iteration 最多生成一个 token，按键和重绘不能被长循环饿死；
- GENERATING 时允许用户中断；
- model load 和首轮 arena allocation 失败时显示有界错误，不半初始化运行；
- context 达到 128 时停止并提示 New Chat，不能写出 KV cache；
- 同一时刻只存在一个活动 AI cell。

## 4. 输入与 cell 行为

首版必须支持：

- 字符插入、退格、光标移动；
- Enter 提交；
- transcript 上下滚动；
- 生成中断；
- New Chat；
- Settings；
- Exit。

具体 TI-Nspire 键位在第一次真机 input probe 后冻结，避免仅凭 Host 键盘臆测。
键位映射必须集中在 Ndless platform 文件，不能散落在 widget 逻辑中。

提交一次用户输入后：

1. 编码为当前 `.ngm` 的字符 token；
2. 添加 USER cell；
3. 将明确的 role/separator prompt token 送入模型；
4. 创建空 AI cell；
5. 每生成一个 token，追加文本并更新 metrics；
6. 遇到停止条件、context 满或用户中断后回到编辑状态。

在后训练格式冻结前，role separator 先作为配置，不硬编码为当前 Shakespeare
词表中不存在的特殊 token。

## 5. telemetry 定义

底栏至少显示：

- `TTFT`：从提交到第一个可见生成 token；
- `tok/s`：只统计生成 token，不把模型加载和用户 prompt prefill 混入；
- `context`：已占用 token / 128；
- `RAM`：模型 blob + arena + UI pools 的当前/峰值 tracked bytes；
- model route：Direct、Distilled 或 Quantized。

字符模型中一个 token 对应一个 vocabulary 字符，因此当前 `tok/s` 也近似
characters/s。以后换 tokenizer 后界面只保留 tokens/s，不误标 characters/s。

同时保留：

```text
prefill tokens/s
decode tokens/s
time to first token
peak tracked heap
model load seconds
```

到设备测量 JSON，避免 UI 为简洁而丢失实验数据。

## 6. 固定容量与内存所有权

首版不在生成循环中调用 `malloc`。启动时分配：

```text
model blob
runtime arena
transcript cell table
transcript UTF-8/text pool
input buffer
framebuffer/backbuffer
```

cell table 和文本 pool 都设固定上限。空间不足时：

- 优先停止追加并显示明确提示；
- 不偷偷写聊天记录到磁盘；
- 不覆盖仍在显示或仍属于模型上下文的内存；
- New Chat 可以一次释放/清零整个 pool。

model blob 与 runtime arena 分离。切换模型必须先销毁旧 runtime，再释放旧 blob，
然后加载新模型；不能同时保留两个 5–6 MiB 模型。

## 7. New Chat 与 Exit 的隐私契约

默认且首版唯一行为：

```text
不保存对话
不恢复上次对话
不创建 transcript/cache 文件
```

New Chat：

1. 停止生成；
2. `ng_runtime_reset` 覆盖 KV 与 scratch；
3. 覆盖 input buffer；
4. 覆盖 transcript text pool；
5. cell count、scroll 和 metrics 归零。

Exit：

1. 执行 New Chat 的全部清理；
2. 释放 transcript/input/backbuffer；
3. 释放 runtime arena；
4. 释放 model blob；
5. 恢复 LCD/input 状态；
6. 返回 Ndless。

测试必须在释放前检查相关 buffer 已归零。`ng_runtime_reset` 已使用 volatile stores，
防止“reset 后立刻 free”被编译器优化掉。

## 8. Settings 首版范围

设置页只包含能诚实支持的项目：

- model file/route；
- greedy 或已实现的 sampling 参数；
- 每次最大生成 token；
- telemetry 显示开关；
- New Chat；
- About/build evidence。

“保存对话”不作为开关：首版始终关闭。未来若要加入，必须另做显式 opt-in、格式
和删除语义设计，不能改变本课的默认隐私保证。

## 9. Host 与真机验收门

Host/headless：

- layout 在极短、极长、换行和滚动 cell 下无越界；
- 状态机不允许两个活动 AI cell；
- context 128 时受控停止；
- 中断生成后仍可继续输入；
- New Chat/Exit 清零对话 buffer；
- 事件队列满时丢弃输入而不破坏状态；
- 无每-token allocation。

Ndless 真机：

- `.tns` 能启动、选择并校验 `.ngm`；
- 可连续完成至少三轮 USER/AI cell；
- 生成时滚动和中断可用；
- TTFT、decode tokens/s 与 tracked peak RAM 有记录；
- New Chat 后 context 回到 0；
- Exit 后回到文档系统且屏幕/按键状态正常；
- 文件系统中没有新增聊天记录。

只有真机门通过后，才能把 `nspire_measurement` 从 pending 改为 measured。
