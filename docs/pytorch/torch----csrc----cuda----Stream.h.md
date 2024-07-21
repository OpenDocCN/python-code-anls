# `.\pytorch\torch\csrc\cuda\Stream.h`

```py
// 如果未定义 THCP_STREAM_INC 宏，则定义 THCP_STREAM_INC 宏，用于避免头文件重复包含
#ifndef THCP_STREAM_INC
// 定义 THCP_STREAM_INC 宏
#define THCP_STREAM_INC

// 包含 CUDAStream 类的头文件，提供 CUDA 流的功能
#include <c10/cuda/CUDAStream.h>

// 包含 Stream 类的头文件，提供 PyTorch 中流的抽象
#include <torch/csrc/Stream.h>

// 包含 Python C API 的头文件，用于与 Python 解释器交互
#include <torch/csrc/python_headers.h>

// 定义 THCPStream 结构体，继承自 THPStream 结构体
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THCPStream : THPStream {
  // CUDAStream 对象，用于管理 CUDA 设备上的流
  at::cuda::CUDAStream cuda_stream;
};

// THCPStreamClass 对象的外部声明
extern PyObject* THCPStreamClass;

// THCPStream_init 函数的声明
void THCPStream_init(PyObject* module);

// 内联函数 THCPStream_Check，用于检查给定对象是否为 THCPStream 类型
inline bool THCPStream_Check(PyObject* obj) {
  // 检查 THCPStreamClass 是否存在且给定对象是否为 THCPStreamClass 的实例
  return THCPStreamClass && PyObject_IsInstance(obj, THCPStreamClass);
}

// 结束 THCP_STREAM_INC 宏定义部分
#endif // THCP_STREAM_INC
```