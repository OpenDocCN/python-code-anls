# `.\pytorch\torch\csrc\jit\passes\restore_mutation.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/core/symbol.h>
// 包含 ATen 库的 symbol.h，用于处理符号定义

#include <c10/util/Exception.h>
// 包含 c10 库的 Exception.h，用于异常处理

#include <torch/csrc/Export.h>
// 包含 torch 库的 Export.h，定义了导出符号

#include <torch/csrc/jit/ir/alias_analysis.h>
// 包含 torch JIT 库的 alias_analysis.h，提供别名分析功能

#include <torch/csrc/jit/ir/ir.h>
// 包含 torch JIT 库的 ir.h，定义了图中的 IR 节点等

namespace torch {
namespace jit {

// 存储激活操作符是否支持类型提升的映射关系的无序映射表
const std::unordered_map<Symbol, bool> activation_type_promotion_mapping = {
    {aten::sigmoid, true},
    {aten::tanh, true},
    {aten::celu, false},
    {aten::elu, false},
    {aten::gelu, false},
    {aten::glu, false},
    {aten::hardshrink, false},
    {aten::hardsigmoid, false},
    {aten::hardswish, false},
    {aten::hardtanh, false},
    {aten::leaky_relu, false},
    {aten::prelu, false},
    {aten::relu6, false},
    {aten::relu, false},
    {aten::rrelu, false},
    {aten::selu, false},
    {aten::silu, false}};

class FunctionalToInplaceRewriter {
 public:
  // 构造函数，接受一个图对象的共享指针
  FunctionalToInplaceRewriter(std::shared_ptr<Graph> graph);

  // 尝试将图中的函数式操作替换为就地操作
  bool FunctionalToInplace(Block* block);

 private:
  // 获取或创建别名分析数据库的私有方法
  AliasDb* getOrCreateAliasDb() {
    // 如果别名分析数据库尚未创建，则创建一个新的并返回
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  // 判断节点是否可以进行就地操作的私有方法
  bool CanBeInplace(Node* node);

  std::unique_ptr<AliasDb> aliasDb_ = nullptr; // 别名分析数据库的唯一指针
  std::shared_ptr<Graph> graph_; // 图对象的共享指针
};

// 将函数式 ATen 激活操作替换为其原位操作的函数
TORCH_API bool FunctionalToInplaceActivation(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
```