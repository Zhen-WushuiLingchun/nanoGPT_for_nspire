# Runtime

这里将实现不依赖 PyTorch 的推理核心：

- `include/`：稳定的公共 C 接口；
- `src/`：可移植算子、模型加载和生成循环；
- `platform/host/`：电脑端参考程序与性能测量；
- `platform/ndless/`：TI-Nspire CX II CAS 平台适配。

设备代码只在 Host 数值对齐通过后开始实现。
