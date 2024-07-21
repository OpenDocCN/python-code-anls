# `.\pytorch\torch\csrc\jit\runtime\interpreter\frame.cpp`

```py
// 包含 Torch 的 JIT 模块中解释器的帧头文件
#include <torch/csrc/jit/runtime/interpreter/frame.h>

// 包含标准库中的原子操作头文件
#include <atomic>

// Torch JIT 解释器命名空间开始
namespace torch::jit::interpreter {

// 定义静态函数 genId，返回一个唯一的帧标识符
/* static */ size_t Frame::genId() {
    // 定义一个静态的原子类型变量 numFrames，用于记录帧的数量
    static std::atomic<size_t> numFrames{0};
    // 使用原子操作增加 numFrames 的值并返回增加前的值作为帧的唯一标识符
    return numFrames.fetch_add(1, std::memory_order_relaxed);
}

} // Torch JIT 解释器命名空间结束
```