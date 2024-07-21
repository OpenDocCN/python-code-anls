# `.\pytorch\torch\csrc\autograd\input_buffer.h`

```
#pragma once

// InputBuffer 类累积一组变量，供函数使用。它实现了逻辑以避免直接修改传入的值
// （重复添加输入将累积结果）。这种行为仅在反向图中需要和使用。

#include <utility>
#include <vector>

#include <c10/core/Stream.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/variable.h>

namespace torch::autograd {

// 定义 InputBuffer 结构体
struct InputBuffer {
  // 显式构造函数，接受一个大小参数并初始化 buffer
  explicit InputBuffer(size_t size) : buffer(size) {}
  // 删除拷贝构造函数
  InputBuffer(const InputBuffer& other) = delete;
  // 默认移动构造函数
  InputBuffer(InputBuffer&& other) = default;
  // 显式构造函数，接受一个 variable_list 并初始化 buffer
  explicit InputBuffer(variable_list&& inputs) : buffer(std::move(inputs)){};
  // 移动赋值运算符
  InputBuffer& operator=(InputBuffer&& other) = default;

  // 在指定索引位置累积变量
  // 可选的 CUDA 流用于确定累积运行的流，并同步添加操作
  TORCH_API void add(
      size_t pos,
      Variable&& var,
      const std::optional<c10::Stream>& opt_producer_stream,
      const std::optional<c10::Stream>& opt_consumer_stream);

  // 返回设备信息
  at::Device device() const;

  // 重载运算符[]，返回指定位置的变量
  Variable operator[](size_t pos) {
    return buffer[pos];
  }

  // 将 InputBuffer 转换为变量列表并销毁原始 InputBuffer
  static std::vector<Variable> variables(InputBuffer&& g);

  // 缓存变量的向量
  std::vector<Variable> buffer;
};

} // namespace torch::autograd
```