# `.\pytorch\aten\src\ATen\cuda\tunable\StreamTimer.cpp`

```py
// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
// 包含 CUDA 运行时的头文件
#include <cuda_runtime.h>

// 包含 C10 CUDA Stream 相关的头文件
#include <c10/cuda/CUDAStream.h>
// 包含 ATen CUDA 异常处理的头文件
#include <ATen/cuda/Exceptions.h>
// 包含 ATen CUDA 可调节性库的计时器类头文件
#include <ATen/cuda/tunable/StreamTimer.h>

// 定义 at::cuda::tunable 命名空间
namespace at::cuda::tunable {

// 构造函数，初始化 StreamTimer 对象
StreamTimer::StreamTimer() {
  // 创建 CUDA 事件 start_
  AT_CUDA_CHECK(cudaEventCreate(&start_));
  // 创建 CUDA 事件 end_
  AT_CUDA_CHECK(cudaEventCreate(&end_));
}

// 析构函数，释放资源
StreamTimer::~StreamTimer() {
}

// 计时开始函数
void StreamTimer::Start() {
  // 同步 CUDA 设备，确保之前的操作完成
  AT_CUDA_CHECK(cudaDeviceSynchronize());
  // 记录当前时间到 start_ 事件，并关联当前 CUDA 流
  AT_CUDA_CHECK(cudaEventRecord(start_, at::cuda::getCurrentCUDAStream()));
}

// 计时结束函数
void StreamTimer::End() {
  // 记录当前时间到 end_ 事件，并关联当前 CUDA 流
  AT_CUDA_CHECK(cudaEventRecord(end_, at::cuda::getCurrentCUDAStream()));
  // 同步等待 end_ 事件完成
  AT_CUDA_CHECK(cudaEventSynchronize(end_));
}

// 获取计时时长函数
float StreamTimer::Duration() {
  float time;
  // 计算 start_ 到 end_ 事件之间的时间差，单位为毫秒，精度为微秒
  AT_CUDA_CHECK(cudaEventElapsedTime(&time, start_, end_));
  return time;
}

} // namespace at::cuda::tunable
```