# Ndless platform

这里保存 TI-Nspire CX II CAS 的文件、屏幕、按键、计时和内存适配。

当前 `make ndless-smoke` 已用安装好的 Ndless SDK 编译、链接并封装 portable
loader、FP32/W4A8 算子和增量 runtime。`main_ndless.c` 只做受控链接 smoke；
它不是最终 UI，也不代表已经完成 CX II 真机测量。

下一步按
[`Lesson 09 UI 设计`](../../../docs/plans/2026-07-28-lesson-09-ndless-chat-ui-design.md)
实现连续 USER/AI cell、逐 token 刷新、telemetry，以及退出时清 cache/不保存对话。
