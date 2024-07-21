# `.\pytorch\torch\csrc\lazy\ts_backend\config.cpp`

```py
// 包含 Torch 框架中延迟计算模块的配置头文件

// 定义一个布尔类型的配置选项 torch_lazy_ts_tensor_update_sync，
// 默认值为 true，控制是否在 _copy_from 操作中使用同步复制
C10_DEFINE_bool(
    torch_lazy_ts_tensor_update_sync,
    true,
    "Use synchronous copy inside _copy_from op");

// 定义一个布尔类型的配置选项 torch_lazy_ts_cuda，
// 默认值为 false，控制是否在 TorchScript 后端中使用 CUDA 设备而不是 CPU
// TODO（whc）我们需要以更有用的方式连接这些标志，也许还要保持 LTC_TS_CUDA 环境的工作？
C10_DEFINE_bool(
    torch_lazy_ts_cuda,
    false,
    "Use cuda device for torchscript backend (instead of CPU)");
```