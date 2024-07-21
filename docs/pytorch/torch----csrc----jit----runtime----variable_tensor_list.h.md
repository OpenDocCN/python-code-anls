# `.\pytorch\torch\csrc\jit\runtime\variable_tensor_list.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

namespace torch::jit {

// torch::jit 命名空间下的变量 tensor 列表结构，继承自 std::vector<at::Tensor>
// 用于标记期望所有 at::Tensor 的位置
struct variable_tensor_list : public std::vector<at::Tensor> {
  // 默认构造函数
  variable_tensor_list() = default;
  // 接受迭代器范围的构造函数
  template <class InputIt>
  variable_tensor_list(InputIt first, InputIt last)
      : std::vector<at::Tensor>(first, last) {}
  // 接受 std::vector<at::Tensor>&& 的构造函数
  explicit variable_tensor_list(std::vector<at::Tensor>&& tensor)
      : std::vector<at::Tensor>(std::move(tensor)) {}
};

} // namespace torch::jit
// 结束 torch::jit 命名空间
```