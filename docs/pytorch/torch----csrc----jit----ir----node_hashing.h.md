# `.\pytorch\torch\csrc\jit\ir\node_hashing.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中的 IR 头文件

namespace torch {
namespace jit {

struct TORCH_API HashNode {
  // 声明结构体 HashNode，用于定义 Node* 类型的哈希函数对象
  size_t operator()(const Node* k) const;
  // 重载 () 运算符，用于计算 Node* 类型对象的哈希值
};

struct TORCH_API EqualNode {
  // 声明结构体 EqualNode，用于定义比较 Node* 类型对象相等性的函数对象
  bool operator()(const Node* lhs, const Node* rhs) const;
  // 重载 () 运算符，用于比较两个 Node* 类型对象是否相等
};

} // namespace jit
} // namespace torch
// 命名空间声明结束
```