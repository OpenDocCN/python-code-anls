# `.\pytorch\torch\csrc\jit\passes\liveness.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <ATen/ATen.h>
// 包含 ATen 库，用于张量操作和计算图构建

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 类型定义，用于表示计算图节点的值

#include <ATen/core/jit_type.h>
// 包含 ATen 库的 JIT 类型定义，用于 JIT 编译时的类型操作

#include <ATen/core/stack.h>
// 包含 ATen 库的堆栈操作，用于计算图节点值的堆栈管理

#include <c10/util/sparse_bitset.h>
// 包含 c10 库的稀疏位集合定义，用于高效管理稀疏位信息

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏定义，用于声明导出接口

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT IR 类定义，用于构建和操作 JIT 中间表示

#include <list>
// 包含 C++ 标准库的链表容器定义

#include <unordered_map>
// 包含 C++ 标准库的无序映射容器定义

#include <vector>
// 包含 C++ 标准库的向量容器定义

namespace torch {
namespace jit {

using SparseBitVector = ::c10::SparseBitVector<256>;
// 定义别名 SparseBitVector，表示一个大小为256的稀疏位集合

// BuildLivenessSets computes "bailout" liveness which is equivalent to
// "{LIVE_IN} or {GEN}" or "{LIVE_OUT} - {KILL}"
// BuildLivenessSets 函数计算 "bailout" 存活性，相当于 "{LIVE_IN} or {GEN}" 或 "{LIVE_OUT} - {KILL}"
TORCH_API std::unordered_map<Node*, std::vector<Value*>> BuildLivenessSets(
    std::shared_ptr<Graph> graph);
// 声明 BuildLivenessSets 函数，接受一个指向图形的共享指针作为参数，返回节点到值向量的无序映射

} // namespace jit
} // namespace torch
```