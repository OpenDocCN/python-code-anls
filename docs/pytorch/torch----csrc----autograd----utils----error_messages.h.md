# `.\pytorch\torch\csrc\autograd\utils\error_messages.h`

```py
#pragma once
// 使用#pragma once确保头文件只被编译一次

#include <sstream>
// 包含头文件<sstream>，提供字符串流的支持

namespace torch {
namespace autograd {
namespace utils {

inline std::string requires_grad_leaf_error(bool requires_grad) {
  // 定义一个内联函数，返回类型为std::string，接受一个bool类型参数requires_grad

  std::ostringstream oss;
  // 创建一个ostringstream对象oss，用于字符串流的输出

  oss << "you can only change requires_grad flags of leaf variables.";
  // 将字符串 "you can only change requires_grad flags of leaf variables." 写入oss

  if (requires_grad == false) {
    // 如果requires_grad为false，执行以下语句块
    oss << " If you want to use a computed variable in a subgraph "
           "that doesn't require differentiation use "
           "var_no_grad = var.detach().";
    // 向oss添加附加说明，指出如何在子图中使用不需要微分的计算变量
  }

  return oss.str();
  // 返回oss中的字符串表示形式
}

} // namespace utils
} // namespace autograd
} // namespace torch
```