# `.\pytorch\torch\csrc\api\src\mps.cpp`

```py
/// 包含 ATen 库的 Context 头文件
/// 包含了 torch MPS 头文件
#include <ATen/Context.h>
#include <torch/mps.h>

/// 定义了 torch::mps 命名空间
namespace torch {
namespace mps {

/// 检查当前设备是否支持 MPS（Metal Performance Shaders）
bool is_available() {
  /// 调用 ATen 库的内部函数，检查是否具有 MPS 支持
  return at::detail::getMPSHooks().hasMPS();
}

/// 设置 MPS 默认生成器的种子
void manual_seed(uint64_t seed) {
  if (is_available()) {
    /// 获取 MPS 默认生成器对象
    auto gen = at::detail::getMPSHooks().getDefaultMPSGenerator();
    {
      // See Note [Acquire lock when using random generators]
      /// 使用互斥锁保护，确保在使用随机生成器时安全地设置种子
      std::lock_guard<std::mutex> lock(gen.mutex());
      /// 设置当前生成器的种子值
      gen.set_current_seed(seed);
    }
  }
}

/// 同步 MPS 设备
void synchronize() {
  /// 调用 ATen 库的内部函数，执行设备同步操作
  at::detail::getMPSHooks().deviceSynchronize();
}

/// 提交 MPS 流
void commit() {
  /// 调用 ATen 库的内部函数，提交当前 MPS 流
  at::detail::getMPSHooks().commitStream();
}

/// 获取 MPS 命令缓冲区
MTLCommandBuffer_t get_command_buffer() {
  /// 调用 ATen 库的内部函数，获取当前 MPS 命令缓冲区
  return at::detail::getMPSHooks().getCommandBuffer();
}

/// 获取 MPS 分发队列
DispatchQueue_t get_dispatch_queue() {
  /// 调用 ATen 库的内部函数，获取当前 MPS 分发队列
  return at::detail::getMPSHooks().getDispatchQueue();
}

} // namespace mps
} // namespace torch
```