# `.\pytorch\torch\csrc\jit\ir\type_hashing.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/core/jit_type.h>
// 包含 ATen 库的 JIT 类型头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 的 IR（Intermediate Representation，中间表示）头文件

namespace torch {
namespace jit {

struct TORCH_API HashType {
  // 结构体 HashType，定义了用于类型指针的哈希函数
  size_t operator()(const TypePtr& type) const;
  // 重载的调用运算符，用于计算给定类型指针的哈希值

  size_t operator()(const c10::ConstTypePtr& type) const;
  // 重载的调用运算符，用于计算给定常量类型指针的哈希值
};

struct EqualType {
  // 结构体 EqualType，定义了用于比较类型指针相等性的函数对象
  bool operator()(const TypePtr& a, const TypePtr& b) const;
  // 重载的调用运算符，用于比较两个类型指针是否相等

  bool operator()(const c10::ConstTypePtr& a, const c10::ConstTypePtr& b) const;
  // 重载的调用运算符，用于比较两个常量类型指针是否相等
};

} // namespace jit
} // namespace torch
```