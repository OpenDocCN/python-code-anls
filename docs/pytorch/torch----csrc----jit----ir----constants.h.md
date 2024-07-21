# `.\pytorch\torch\csrc\jit\ir\constants.h`

```py
#pragma once
// 包含 ATen 库中的相关头文件，用于处理 IR 中的常量
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
// 导入 Torch 的导出声明
#include <torch/csrc/Export.h>
// 包含 Torch 的源范围定义
#include <torch/csrc/jit/frontend/source_range.h>
// 包含 Torch 的作用域定义
#include <torch/csrc/jit/ir/scope.h>

// helpers for handling constants in the IR
// - create constant nodes from ints, floats, complex, intlist, Tensors, and
// other types
// - implement primitive constant ops.
// 定义命名空间 torch::jit 下的常量处理辅助函数
namespace torch {
namespace jit {

// 使用 ATen 库中的 IValue 类型
using ::c10::IValue;

// 声明 Graph 和 Value 结构体
struct Graph;
struct Value;

// 当插入常量无法编码为图时抛出的异常
struct TORCH_API constant_not_supported_error : public std::runtime_error {
  using runtime_error::runtime_error;
};

// 在图中插入常量节点的函数声明
TORCH_API Value* insertConstant(
    Graph& g,
    const IValue& val,
    std::optional<SourceRange> loc = c10::nullopt,
    std::optional<ScopePtr> scope = c10::nullopt);

// 注意: 推荐使用 g.insertConsant(val, loc)，它与此函数功能相同
// 此函数仅在此处声明/定义，因为其实现与 constants.cpp 中的 prim::Constant
// 实现密切相关。
// 如果 IValue 类型无法作为常量插入，则返回 c10::nullopt
TORCH_API std::optional<Value*> tryInsertConstant(
    Graph& g,
    const IValue& val,
    std::optional<SourceRange> loc = c10::nullopt,
    std::optional<ScopePtr> scope = c10::nullopt);

////////////////////////////////////////////////////////////////////////////////
// Helper for retrieving constants
////////////////////////////////////////////////////////////////////////////////

// 尝试将（可能是常量的）Value* 转换为解释器值（IValue）
// 如果 Value* 不是常量，则返回 c10::nullopt
TORCH_API std::optional<IValue> toIValue(const Value* v);

// 如果值是常量，则尝试使用与解释器相同的规则将其转换为类型 T
template <typename T>
std::optional<T> constant_as(const Value* v) {
  if (auto ivalue = toIValue(v)) {
    return ivalue->to<T>();
  }
  return c10::nullopt;
}

} // namespace jit
} // namespace torch
```