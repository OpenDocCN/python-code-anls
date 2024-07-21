# `.\pytorch\c10\cuda\CUDAGraphsC10Utils.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/cuda/CUDAStream.h>
// 引入 CUDAStream 头文件，提供 CUDA 流的相关功能

#include <iostream>
// 引入标准输入输出流库，用于输出和错误处理

#include <utility>
// 引入实用工具库，提供 std::pair 等实用工具

// CUDA Graphs utils used by c10 and aten.
// aten/cuda/CUDAGraphsUtils.cuh adds utils used by aten only.
// 命名空间说明，定义了 c10::cuda 命名空间，包含 CUDA 图的实用工具

namespace c10::cuda {

using CaptureId_t = unsigned long long;
// 使用 CaptureId_t 定义了无符号长整型别名，用于表示捕获 ID

// first is set if the instance is created by CUDAGraph::capture_begin.
// second is set if the instance is created by at::cuda::graph_pool_handle.
// 使用 std::pair 定义了 MempoolId_t 别名，用于表示由 CUDAGraph::capture_begin 或 at::cuda::graph_pool_handle 创建的实例

// RAII guard for "cudaStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
// CUDAStreamCaptureModeGuard 类的定义，用于管理 "cudaStreamCaptureMode" 的 RAII 守卫

struct C10_CUDA_API CUDAStreamCaptureModeGuard {
  CUDAStreamCaptureModeGuard(cudaStreamCaptureMode desired)
      : strictness_(desired) {
    // 构造函数，设置严格性并交换线程的捕获模式
    C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }
  // 析构函数，恢复之前的捕获模式
  ~CUDAStreamCaptureModeGuard() {
    C10_CUDA_CHECK_WARN(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  cudaStreamCaptureMode strictness_;
  // 成员变量，存储捕获模式
};

// Protects against enum cudaStreamCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
// 静态断言，保护免受 enum cudaStreamCaptureStatus 实现更改的影响
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) == 0,
    "unexpected int(cudaStreamCaptureStatusNone) value");
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive) == 1,
    "unexpected int(cudaStreamCaptureStatusActive) value");
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated) == 2,
    "unexpected int(cudaStreamCaptureStatusInvalidated) value");

// 枚举类型 CaptureStatus，表示 CUDA 图捕获的状态
enum class CaptureStatus : int {
  None = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone),
  Active = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive),
  Invalidated = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated)
};

// 重载流输出操作符，输出 CaptureStatus 枚举值的字符串表示
inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "cudaStreamCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "cudaStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "cudaStreamCaptureStatusInvalidated";
      break;
    default:
      // 异常处理，输出未知的 CUDA 图 CaptureStatus
      TORCH_INTERNAL_ASSERT(
          false, "Unknown CUDA graph CaptureStatus", int(status));
  }
  return os;
}

// Use this version where you're sure a CUDA context exists already.
// currentStreamCaptureStatusMayInitCtx 函数定义，获取当前 CUDA 流的捕获状态
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  // 存储捕获状态的变量
  cudaStreamCaptureStatus is_capturing{cudaStreamCaptureStatusNone};
  // 检查当前 CUDA 流是否正在捕获，获取捕获状态
  C10_CUDA_CHECK(
      cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
  // 返回捕获状态
}

} // namespace c10::cuda
// 命名空间结束
```