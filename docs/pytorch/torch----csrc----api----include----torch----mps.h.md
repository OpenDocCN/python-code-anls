# `.\pytorch\torch\csrc\api\include\torch\mps.h`

```py
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <torch/csrc/Export.h>
// 引入 Torch 库导出的头文件

#include <cstddef>
#include <cstdint>
// 引入标准 C++ 库头文件

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
// 如果是 Objective-C 环境，引入 Foundation 和 Metal 框架头文件
using MTLCommandBuffer_t = id<MTLCommandBuffer>;
using DispatchQueue_t = dispatch_queue_t;
// 定义 Metal 相关类型的别名
#else
using MTLCommandBuffer_t = void*;
using DispatchQueue_t = void*;
// 如果非 Objective-C 环境，定义 Metal 相关类型的别名为 void*
#endif

namespace torch {
namespace mps {

/// Returns true if MPS device is available.
// 如果 MPS 设备可用，返回 true
bool TORCH_API is_available();

/// Sets the RNG seed for the MPS device.
// 设置 MPS 设备的随机数生成器种子
void TORCH_API manual_seed(uint64_t seed);

/// Waits for all streams on the MPS device to complete.
/// This blocks the calling CPU thread by using the 'waitUntilCompleted()'
/// method to wait for Metal command buffers finish executing all the
/// encoded GPU operations before returning.
// 等待 MPS 设备上所有流执行完成。
// 通过调用 'waitUntilCompleted()' 方法阻塞当前 CPU 线程，
// 等待 Metal 命令缓冲区完成所有编码的 GPU 操作后再返回。
void TORCH_API synchronize();

/// Submits the currently active command buffer to run on the MPS device.
// 提交当前活跃的命令缓冲区以在 MPS 设备上运行
void TORCH_API commit();

/// Get the current command buffer to encode the Metal commands.
// 获取当前的命令缓冲区用于编码 Metal 命令
MTLCommandBuffer_t TORCH_API get_command_buffer();

/// Get the dispatch_queue_t to synchronize encoding the custom kernels
/// with the PyTorch MPS backend.
// 获取 dispatch_queue_t 以同步编码自定义内核与 PyTorch MPS 后端
DispatchQueue_t TORCH_API get_dispatch_queue();

} // namespace mps
} // namespace torch


这段代码是 C++ 头文件，定义了一些与 Metal Performance Shaders (MPS) 相关的函数和类型别名，并提供了相应的注释说明其作用和功能。
```