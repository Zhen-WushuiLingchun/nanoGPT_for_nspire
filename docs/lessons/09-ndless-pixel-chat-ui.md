# Lesson 09：Ndless 像素对话界面、隐私生命周期与真机部署

这一课把 Lesson 08 的“一次输入一个 token 的 C runtime”装进一个真正可交互的
TI-Nspire 应用：

```text
键盘 -> input buffer -> prompt prefill -> KV cache
                                  |
                                  v
                         one-token decode
                                  |
                                  v
RGB565 framebuffer <- USER / AI cells <- UTF-8 token
```

现在已经完成：

1. 固定容量的 input、transcript、cell 和 pending-token 状态；
2. prompt prefill 与 autoregressive decode 状态机；
3. 320×240 RGB565 像素界面；
4. TTFT、tokens/s、context 和 tracked RAM 显示；
5. New Chat/Exit 的 volatile zeroing；
6. Host headless fixture、真实 `.ngm` 集成测试；
7. 完整 Ndless ARM compile、link 和 `.tns` package。

生成的应用为：

```text
dist/nanogpt-chat.tns       59,310 bytes
dist/model.ngm.tns       6,036,544 bytes  # 默认 Quantized-Small
```

本轮也检测到了真实 CX II，并成功读取过设备根目录；第一次 6 MiB 多文件同步在
约 50.9 秒遇到 LibUSB error。失败后设备仍能重新枚举，但 TI 文件服务握手继续
超时，需要物理拔插 USB 后才能审计临时文件并重传。因此当前证据是
Host 通过、ARM package 通过、真机上传未确认，不把它写成“已经在计算器运行”。

## 1. 这是不是一个 ChatGPT

界面看起来像对话：

```text
[USER] Explain causal attention.
[AI]   ...

[USER] Why triangular?
[AI]   ...
```

但当前模型仍是 Tiny Shakespeare 字符续写模型。USER/AI 是界面 metadata，
不是模型理解的 role token；模型实际看到的是连续字符和换行：

```text
Explain causal attention.\n...
Why triangular?\n...
```

所以必须区分三层：

| 层 | 当前已经有什么 | 当前还没有什么 |
|---|---|---|
| UI | 多轮 cell、输入、滚动、实时输出 | 触摸菜单、完整设置页 |
| 推理 runtime | KV cache、逐 token decode、W4A8 | instruction/chat 语义 |
| 模型 | Shakespeare 字符分布 | 物理解释能力、服从问题的能力 |

把 completion model 套上聊天界面，不会自动把它变成 instruction model。
这个界面先解决部署、交互、内存和生命周期；以后做 physics corpus 后训练或蒸馏时，
同一个前端可以原样复用。

## 2. prefill 与 decode 为什么是两个阶段

用户按 Enter 后，模型不能直接“凭整段字符串”生成。字符 tokenizer 要把输入逐个
映射成 token，再依次送入 Transformer：

```text
PREFILL:
    input token 0 -> forward -> KV[0]
    input token 1 -> forward -> KV[1]
    ...
    newline       -> forward -> KV[n]

GENERATING:
    argmax(logits)
    append visible token
    generated token -> forward -> KV[n+1]
    repeat
```

prefill 的作用是建立 prompt 的 KV cache，并在最后一个 prompt token 后得到第一组
next-token logits。decode 才是在这些 logits 上选 token、显示 token，并把它写回
context。

这也是 TTFT 与 decode tokens/s 必须分开的原因：

```text
TTFT = 用户提交
       + prompt prefill
       + 第一次 token 选择/forward

decode tokens/s = 第一个可见 token 之后的持续生成速度
```

若把 prefill 混进 tokens/s，短 prompt 和长 prompt 的数字不可比较；若把模型加载
也混进去，第一轮与第二轮又不可比较。

## 3. 为什么 event loop 每次最多推进一个 token

一个最直接但不好的写法是：

```c
while (generated < 32) {
    forward_one_token();
}
```

在桌面上它只是暂时卡住窗口；在计算器上它会让按键、中断和 framebuffer 更新全部
饿死。现在主循环的结构是：

```text
poll one semantic input event
handle edit / send / scroll / new chat / exit

if PREFILL or GENERATING:
    ng_chat_step()       # 最多一个 forward

if dirty:
    render RGB565
    lcd_blit

yield 10 ms
```

因此生成中的 AI cell 会逐字符出现，Esc 可以在 token 边界中断，Menu 可以执行
New Chat。每次 `ng_chat_step` 使 runtime context 最多增长 1，集成测试直接检查了
这个不变量。

## 4. 固定容量不是“省事”，而是部署契约

`ng_chat` 不在 token loop 内调用 `malloc`。容量在编译时冻结：

| 区域 | 容量 |
|---|---:|
| input | 192 bytes |
| transcript text pool | 4,096 bytes |
| cell table | 24 cells |
| pending prefill queue | 256 token IDs |
| runtime context | 来自模型，当前为 128 tokens |

cell 不各自拥有字符串。它只记录：

```c
role
text_offset
text_length
```

真正的 UTF-8 bytes 顺序放在一个 transcript pool 中：

```text
offset 0                  offset 25
| USER text ............. | AI text ................. |
```

这种 arena-like 设计有三个好处：

1. 内存上限在启动前已知；
2. 生成时追加 token 不产生 heap 碎片；
3. New Chat 可以一次覆盖整个 pool。

空间不足时，提交或追加返回 `NG_CHAT_FULL`，界面进入明确错误状态；它不会越界、
覆盖旧 cell，也不会偷偷把 transcript 写到文件。

## 5. input 编辑为什么仍要移动内存

input 是短小的连续 byte array，光标插入使用：

```text
"ac\0"
   ^

memmove tail right by one

"abc\0"
  ^
```

这里 `memmove` 是正确的，因为 source 与 destination 重叠。退格做相反方向移动。
192-byte 上限使最坏移动成本很小，而连续 buffer 让提交、清零和显示都更简单。

首版硬件键盘输入只接受可显示 ASCII。模型词表仍允许 UTF-8 单字符 token；生成端
会保留精确 UTF-8 bytes。独立实现的 5×7 首版字体把小写显示为相应大写，并把未有
字形的 Unicode scalar 显示为一个 `?`，但 transcript 中的原始 bytes 不会被改写。

## 6. token lookup 为什么不能直接用 ASCII 数值

字符 tokenizer 不代表：

```text
token_id == ASCII code
```

token ID 由训练词表顺序决定。例如 tiny 测试模型中：

```text
0 -> "\n"
1 -> "a"
2 -> "é"
```

提交前，`ng_chat_submit` 对每个输入字符在 `.ngm` vocabulary 中做 exact byte
match。任一字符不存在时，整个提交保持 transaction-like：

- input 不清空；
- USER/AI cell 不创建；
- runtime context 不前进。

只有所有 token 都能编码、context 和 pools 都有空间时，才同时创建 USER cell、
空 AI cell 和 prefill queue。

## 7. greedy decode 与停止条件

当前固定使用：

```text
next_token = argmax(logits)
```

这是为了让 PyTorch、Host C 和 Ndless 的离散序列容易对齐。停止条件为：

- 达到本轮 max generation tokens；
- context 达到模型 `block_size`；
- 连续生成两个 newline；
- 用户中断；
- runtime 或 transcript capacity error。

以后可以加入 temperature/top-k，但那需要固定 PRNG、seed、浮点语义和跨平台采样
测试，不能只在 UI 上加一个看起来存在的选项。

## 8. 320×240 为什么仍使用完整 backbuffer

CX II 屏幕是 320×240。RGB565 每像素 2 bytes：

```text
320 * 240 * 2 = 153,600 bytes
```

界面在 off-screen buffer 中完成，再一次 `lcd_blit`：

```text
+--------------------------------------------------+
| NANOGPT // N-SPIRE      47/128      QUANT W4A8  |
+--------------------------------------------------+
| USER                                             |
| Explain causal attention.                       |
|                                                  |
| AI                                               |
| Causal attention ...                         [ ] |
+--------------------------------------------------+
| > input_cursor_                                  |
+--------------------------------------------------+
| 8.7 T/S | TTFT:910MS | RAM:8.2M                 |
+--------------------------------------------------+
```

视觉系统只使用少量颜色：

- graphite/navy 背景；
- amber USER 和 input；
- phosphor-mint AI、ready 和 streaming cursor；
- coral error；
- 一像素直角边框。

renderer 有自己的 clip rectangle。cell 再长也只能画在 transcript viewport，
不能覆盖 input/footer。Host fixture 覆盖 ready、conversation、generating 和 error
四种状态；RGB565 framebuffer 还用 FNV-1a golden hash 检查确定性。

## 9. telemetry 到底测了什么

底栏当前显示：

```text
decode tokens/s
TTFT milliseconds
tracked RAM MiB
```

header 显示：

```text
context used / block_size
model storage route
```

对 Quantized-Small：

```text
model blob       6,036,544
runtime arena    2,378,624
framebuffer        153,600
chat state       sizeof(ng_chat)
--------------------------------
tracked total    8,568,768 + sizeof(ng_chat)
```

这个数字不是 OS RSS，也不含 allocator metadata、C library 内部状态和所有静态
对象。因此界面明确称 tracked memory。ARM ELF 还单独记录：

```text
text   74,372 bytes
data    7,400 bytes
bss    12,276 bytes
```

Ndless 公共 `gettimeofday` 在当前 SDK 只有 1 秒分辨率。它能诚实测量很慢的真机
生成和长窗口平均速度，但亚秒 token 会显示 0。现在没有为了得到好看的数字而直接
写未验证的 timer MMIO；真机恢复后先做独立、可退出的高分辨率 timer probe，再把
其安全实现接入 UI。

## 10. New Chat 为什么同时是正确性与隐私操作

KV cache 决定下一 token 会看到什么。只清屏、不清 KV 会造成：

```text
屏幕：空白新对话
模型：仍然看到旧对话
```

这既是语义 bug，也是隐私 bug。现在 New Chat 顺序为：

1. `ng_runtime_reset` 使用 volatile stores 覆盖完整 arena；
2. 覆盖 input；
3. 覆盖 transcript pool；
4. 覆盖 cell table；
5. 覆盖 pending tokens；
6. 清空 metrics、scroll 和活动 logits pointer；
7. 保留只读 model/runtime attachment 与启动时 memory accounting。

Host 测试重复执行 8 轮：

```text
submit -> prefill -> generate -> new chat
```

每一轮后都逐 byte 检查 arena、input、transcript、cells 和 pending tokens 为 0，
并检查 tracked peak 不增长。

Exit 再做：

1. New Chat 的全部清理；
2. 把 backbuffer 置零并 blit；
3. `lcd_init(SCR_TYPE_INVALID)` 恢复显示模式；
4. free backbuffer；
5. free runtime arena；
6. free model blob；
7. 返回 Ndless。

代码中没有 transcript open/write 路径，应用不保存、也不恢复聊天记录。

## 11. Ndless 层只负责哪些事情

portable 层不知道 TI 键矩阵、LCD 或 Documents 路径。Ndless adapter 只负责：

- `lcd_init(SCR_320x240_565)` 与 `lcd_blit`；
- key edge detection；
- Shift 字符映射；
- `gettimeofday` 和 `msleep`；
- framebuffer ownership；
- `.ngm.tns` 文件位置；
- 显示模式恢复。

当前键位：

| 键 | 行为 |
|---|---|
| 字母/数字/常用标点 | 输入字符 |
| Shift + 字符 | 大写或 shifted 标点 |
| Left / Right | 移动输入光标 |
| Del | 退格 |
| Enter | 提交 |
| Up / Down | transcript 滚动 |
| Menu | New Chat |
| Esc | 生成中断；空闲时退出 |
| Ctrl + Esc | 退出 |

按键使用 rising-edge debounce；按住一个键不会以 event-loop 速度无限重复。
首版没有 touchpad pointer，避免方向键和 touchpad contact 在 CX II 上产生重叠事件。

## 12. 为什么模型文件叫 `.ngm.tns`

TI Documents 浏览器要求可传输文档使用 `.tns` 后缀。模型内容仍是 Lesson 08 的
`.ngm`，只是设备文件名写成：

```text
model.ngm.tns
```

loader 不依赖扩展名，而是验证：

```text
magic
schema
lengths and offsets
CRC32
vocabulary UTF-8
tensor shapes/storage
model + arena memory limit
```

应用依次寻找：

```text
model.ngm.tns
quantized-small.ngm.tns
distilled-small.ngm.tns
direct-small.ngm.tns
```

先找应用同目录，再找 Documents 的 `/nanoGPT/`。推荐只放一个标准名
`model.ngm.tns`；切换路线时原子替换这个文件，避免启动时同时加载两个模型。

## 13. 构建部署 bundle

Host：

```powershell
cmake -S . -B build/host -G "Visual Studio 17 2022" -A x64
cmake --build build/host --config Release
ctest --test-dir build/host -C Release --output-on-failure
```

Ndless：

```bash
export NDLESS_SDK="$HOME/.phy-nspire/Ndless/ndless-sdk"
export _NDLESS_TOOLCHAIN_PATH="$HOME/.phy-nspire/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi/bin"
export PATH="$NDLESS_SDK/bin:$_NDLESS_TOOLCHAIN_PATH:/usr/bin:/bin"
make ndless-chat
```

选择量化模型作为默认部署模型：

```powershell
Copy-Item artifacts\lesson08-export\quantized-small.ngm `
  dist\model.ngm.tns
```

设备目录保持独立：

```text
Documents/
└── nanoGPT/
    ├── nanogpt-chat.tns
    └── model.ngm.tns
```

若使用已经修复 CX II 大文件/ACK/读回问题的 `phy-nlinkctl`：

```bash
phy-nlinkctl sync \
  --upload dist/nanogpt-chat.tns /nanoGPT/nanogpt-chat.tns \
  --upload dist/model.ngm.tns /nanoGPT/model.ngm.tns
```

`sync` 的 `.upload -> readback SHA-256 -> atomic rename` 很重要。模型是 6 MiB；
不能因为 upload API 返回成功就假设设备端完整。发生 LibUSB error 后先恢复物理
连接并 `ls /nanoGPT`，再决定 fresh sync；远端状态未知时不要直接使用
`--reuse-temporary`。

本轮本地 bundle：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `nanogpt-chat.tns` | 59,310 | `d1590fb4…813d8f8` |
| `model.ngm.tns` | 6,036,544 | `87882cae…5647c9` |

## 14. 哪些结论已经成立

已经成立：

- fixed-capacity chat state 可工作；
- prefill/decode 每次 loop 最多前进一个 token；
- tiny `.ngm` 的真实 runtime session 可重复；
- 320×240 renderer 不越界且 visual fixture 已审查；
- New Chat/Shutdown 私有 buffer 清零；
- 完整 chat program 能被 ARM 工具链以 warnings-as-errors 编译、链接和封装；
- 外部 Quantized-Small bundle 已在 Host 准备完成。

尚未成立：

- `nanogpt-chat.tns` 已在当前 CX II 成功启动；
- calculator-side model open 已通过；
- CX II 的真实 TTFT/tokens/s；
- calculator-side peak heap；
- 退出后 LCD/按键恢复已由人眼确认；
- 设备端 `/nanoGPT` 没有失败上传留下的临时文件。

这些边界记录在
[`experiments/lesson09-chat-ui.json`](../../experiments/lesson09-chat-ui.json)。

## 15. 真机恢复后的验收顺序

物理拔插 USB 后：

1. `ls /nanoGPT`，记录正式、`.upload` 和 `.previous` 文件；
2. fresh `sync` 两个 bundle 文件；
3. 对两个文件完整读回 SHA-256；
4. 从 Documents 打开 `nanoGPT/nanogpt-chat.tns`；
5. 输入一轮短 ASCII prompt；
6. 生成中按 Esc；
7. 再输入第二轮，确认 cell 连续；
8. Menu New Chat，确认 context 回到 0；
9. Ctrl+Esc 退出，确认 Documents 屏幕正常；
10. 重新启动，确认没有恢复上次 transcript；
11. 记录至少 32 个 decode token 的 wall time 与 tracked RAM；
12. 再决定是否启用高分辨率 timer probe。

完成这组验收后，Lesson 09 的 `physical_device` 才能从 pending 改成 measured。
下一阶段则可以一边优化 W4A8 ARM kernel，一边准备真正的物理解释后训练数据；UI
与模型能力仍保持两个可以独立验证的层。
