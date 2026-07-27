# Lesson 08：统一模型格式、Host C 与 Ndless 对齐设计

日期：2026-07-28  
状态：已按既有九课路线开始实施

## 1. 本课目标

把 Lesson 05–07 的三个 PyTorch 候选变成同一种可验证的二进制模型文件，并让
不依赖 PyTorch 的纯 C runtime 完成增量推理：

```text
Direct-Small FP32
Distilled-Small FP32
Quantized-Small packed INT4 + dynamic INT8 activation
```

Host C 是数值裁判和调试环境；同一 portable core 必须通过 Ndless ARM
交叉编译。真机加载时间、字符速度和 tracked heap 仍在 Lesson 09 实测，不能用
Host 或交叉编译成功代替。

## 2. 三种实现路线

### 路线 A：只实现 FP32 C

优点是最快得到 Direct/Distilled 对齐；缺点是完全没有关闭 Quantized-Small
“直接消费整数权重”的门，因此不能算本课完成。

### 路线 B：直接写完整 W4A8 runtime

可以最早碰到真实整数误差，但文件格式、加载器、Transformer 算子和量化 kernel
同时变化，出现偏差时难以定位是序列化、普通算子还是量化造成。

### 路线 C：统一格式，FP32 oracle 后接 W4A8

采用这条路线：

1. 先冻结一种同时容纳 FP32 和 INT4 tensor 的格式；
2. 用 FP32 Direct-Small 对齐 LayerNorm、attention、GELU、KV cache 和 logits；
3. 再替换矩阵 kernel 为直接读取 nibble 的 W4A8 路径；
4. C W4A8 先对齐 Python W4A8 reference，再比较 W4A8 与现有 W4A32
   dequantized reference 的额外质量损失。

这样误差来源保持分层，两个 runtime 仍共享模型加载、attention、KV cache、采样
和内存规划代码。

## 3. 统一模型文件 v1

文件扩展名使用 `.ngm`；Lesson 09 向计算器传输时再决定是否包装或改名为
`.ngm.tns`。格式固定为 little-endian：

```text
128-byte header
64-byte fixed tensor entries
UTF-8 character vocabulary
64-byte-aligned tensor payloads
```

header 至少包含：

- 八字节 magic `NGNSP001`；
- schema version、header size 和 endian marker；
- flags：tied embedding、bias、tanh GELU；
- 完整文件长度；
- header CRC32 与 header 之后全部 payload 的 CRC32；
- tensor table、vocabulary 和 data 区的 offset/length；
- `vocab_size/block_size/n_layer/n_head/n_embd/mlp_ratio`；
- 模型 storage route；
- weight/activation 量化方案和 group size。

每个 tensor entry 包含稳定 tensor ID、storage type、rank、四维 shape、group
metadata、data/aux offset 和 bytes、logical element count。固定 GPT v1 使用
tensor ID 而不是运行时字符串查找；Python manifest 仍记录可读名称。

FP32 tensor 保存 little-endian float32。INT4 tensor 的 data 是 low-nibble-first
packed signed values，aux 是 little-endian FP32 group scales。一维 LayerNorm
weight 仍保存 FP32。tied `lm_head.weight` 只保存
`token_embedding.weight` 一份，并由 header flag 恢复别名。

exporter 必须拒绝：

- 非固定的 bias/tie/GELU 配置；
- 缺失、多余或 shape 不匹配的 tensor；
- 非有限 FP32 值或非法 INT4 metadata；
- vocabulary 数量不匹配或空 token；
- 不受支持的 checkpoint schema/route；
- 导出后超过 6 MiB 的文件。

加载器在返回任何 tensor pointer 前验证：

- magic、版本、端序和两个 CRC32；
- 声明 file size 等于真实 file size；
- 所有 `offset + length` 无整数溢出且在文件内；
- table/vocab/data 区不重叠、不倒序；
- tensor ID 唯一、数量和固定架构一致；
- shape、storage、group size 和 byte count 精确匹配；
- 计算出的 model blob + arena 不超过调用者给定上限。

## 4. C runtime 数据流

生成使用逐 token KV cache：

```text
token id + absolute position
  -> embedding lookup
  -> per block:
       LayerNorm
       QKV projection
       append K/V to cache
       causal attention over cache[0:position]
       output projection + residual
       LayerNorm
       MLP GELU + residual
  -> final LayerNorm
  -> tied vocabulary projection
  -> logits
```

cache 使用 FP32，静态大小：

```text
2 * layers * context * width * 4 bytes
```

Direct/Distilled 为 `655,360 bytes`；Quantized 为 `2,359,296 bytes`。
所有 scratch 从一个有界 arena 一次预分配，生成循环不得继续 malloc。

当前 Python sampler 在超过 context 后会裁剪最近 128 token并重新编号 position。
为了保持完全一致，C 在 cache 填满后若继续生成，会用最近 128 token 重建 cache。
速度验收另外记录未触发重建的前 64 个生成 token，避免把两种语义混在一起。

## 5. FP32 kernel

可移植 C11 标量参考实现包括：

- FP32 embedding lookup 和 row-major matvec；
- bias-free LayerNorm，epsilon `1e-5`；
- fused QKV 的切片；
- scaled causal self-attention；
- stable softmax（先减最大值）；
- tanh approximate GELU；
- residual add；
- tied vocabulary projection。

PyTorch 对齐固定使用 CPU、`model.eval()` 和 batch size 1，避免 CUDA TF32
进入 oracle。预注册门：

```text
last-token logits max absolute error <= 2e-4
last-token logits RMSE               <= 5e-5
64-token greedy sequence exact match
future-token/cache isolation         exact by construction tests
```

若失败，先定位第一个偏离的中间 tensor，不能看到结果后放宽门槛。

## 6. INT4/W4A8 kernel

Quantized-Small 不允许在 load 时展开完整 FP32 matrix。矩阵乘法沿最后一维每
64 个值分组：

```text
x_scale[g] = max(abs(x_group)) / 127
x_q[g]     = round(x_group / x_scale[g]) clipped to [-127, 127]

dot_i,g = int32_sum(x_q[g] * w_int4[i,g])
out[i] += dot_i,g * x_scale[g] * w_scale[i,g]
```

输入 activation 的量化结果对同一个 matvec 的所有 output rows 复用。INT4
nibble 在 dot loop 中按需解码；只允许 `O(input_width)` 的 INT8 scratch 和
`O(group_count)` 的 FP32 activation scales，不允许 `O(parameter_count)` 的
反量化 scratch。

embedding lookup 直接解码所选行；LayerNorm、attention、softmax、GELU、
residual 和 KV cache 保持 FP32。因此“W4A8/int32”准确描述主要 weight
matvec，而不是声称整个 Transformer 没有浮点运算。

对齐分两层：

```text
C W4A8 vs Python W4A8 reference:
    logits max absolute error <= 2e-4
    logits RMSE <= 5e-5
    64-token greedy exact match

W4A8 vs Lesson 07 W4A32 reference:
    validation loss extra degradation <= 0.02
    final validation loss < Direct-Small loss 1.4997899746894836
```

第二层门可能失败；失败也保留结果，不能把 W4A32 指标冒充真实 C 质量。

## 7. Host 与 Ndless 边界

portable core 只能依赖 C11、`stdint.h/stddef.h/string.h/math.h` 和一个很窄的
allocator/file boundary。不得使用：

- Windows API；
- POSIX mmap、pthread 或未界定的 filesystem 行为；
- BLAS/OpenMP/SIMD intrinsics；
- C++ runtime；
- 无界递归或生成期间动态增长的容器。

Windows Host 使用 CMake + MSVC 构建。WSL 使用已安装的：

```text
Ndless SDK:
  /home/hydro/.phy-nspire/Ndless/ndless-sdk

Arm GNU Toolchain:
  /home/hydro/.phy-nspire/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi
```

Lesson 08 的 Ndless gate 是：portable loader/operators 与最小 device entry
能够用 `nspire-gcc` 编译、链接并由 `genzehn/make-prg` 生成 `.tns`。这只证明
ABI/依赖可移植，不证明真机可运行或更快。

## 8. 测试和证据

测试分层：

1. Python binary-format 单元测试；
2. deterministic tiny model round trip；
3. malformed files：bad magic/version/CRC、truncation、overflow、overlap、
   duplicate/missing tensor；
4. C operator golden tests；
5. C loader 对 Python fixture；
6. Direct/Distilled real artifact logits 与 greedy 对齐；
7. INT4 direct-packed 与 Python W4A8 对齐；
8. 6 MiB、24 MiB 静态内存账本；
9. MSVC CTest；
10. Ndless ARM compile/link/package smoke。

大 `.ngm`、probe 和 build 产物留在 `artifacts/`/`build/`；进入 Git 的只有代码、
小型 malformed fixtures、计划、课程和有 SHA-256 的结果摘要。

## 9. 非目标

本课不做：

- 真机速度或 tracked heap 声明；
- GUI、触摸板输入和字体渲染；
- ARM 汇编、NEON 或多线程优化；
- 中文 tokenizer；
- 蒸馏后量化组合实验；
- 修改已冻结的三模型训练结果。

这些内容分别属于 Lesson 09 或后续组合优化。
