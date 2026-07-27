# Runtime

这里是不依赖 PyTorch 的 portable C11 推理核心：

- `include/`：稳定的公共 C 接口；
- `src/`：严格 `.ngm` loader、FP32/W4A8 算子和增量 KV runtime；
- `platform/host/`：电脑端参考程序与性能测量；
- `platform/ndless/`：TI-Nspire CX II CAS 平台适配。

当前 Direct/Distilled FP32 与 Quantized packed W4A8 均已通过 Host 数值对齐。
Ndless 工具链也已编译、链接并封装完整 runtime；真机 UI、heap 和速度测量进入
Lesson 09。
