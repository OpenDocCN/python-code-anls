# `.\pytorch\aten\src\ATen\cuda\tunable\StreamTimer.h`

```
// 原始 TunableOp 来自 onnxruntime。
// 可在以下链接找到原始代码：
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// 版权所有 (c) Microsoft Corporation.
// 使用 MIT 许可证授权。

// 将 TunableOp 转换为 PyTorch 的适配
// 版权所有 (c) Advanced Micro Devices, Inc.

#pragma once

#include <cuda_runtime.h>  // CUDA 运行时头文件

#include <ATen/cuda/tunable/Tunable.h>  // 引入 PyTorch 中的 Tunable 类

namespace at::cuda::tunable {

// StreamTimer 类，实现 ITimer 接口
class StreamTimer : public ITimer {
  public:
    StreamTimer();  // 构造函数声明

    virtual ~StreamTimer();  // 虚析构函数声明

    void Start() override;  // 开始计时，重写 ITimer 接口中的方法

    void End() override;  // 结束计时，重写 ITimer 接口中的方法

    float Duration() override;  // 获取计时持续时间，重写 ITimer 接口中的方法

  private:
    cudaEvent_t start_;  // CUDA 事件，用于开始计时
    cudaEvent_t end_;    // CUDA 事件，用于结束计时
};

} // namespace at::cuda::tunable
```