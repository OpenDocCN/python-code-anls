# `.\pytorch\torch\csrc\jit\passes\symbolic_shape_analysis.h`

```
#pragma once

#include <torch/csrc/Export.h>  // Torch导出API定义
#include <torch/csrc/jit/ir/ir.h>  // Torch JIT中间表示(IR)的头文件
#include <unordered_map>  // 无序映射容器
#include <utility>  // 实用工具组件
#include <variant>  // 可变类型相关

namespace torch {
namespace jit {

// 注意：此功能尚未完成，不稳定，慎用

// 在给定图上执行形状传播
TORCH_API void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph);

// 注意：此功能尚未完成，不稳定，慎用
// 从[beg, end)范围内尝试传播形状，并构建一个图，用于计算所有在[beg, end)范围内仍可执行的符号形状
struct ShapeComputeGraphMapping {
  ShapeComputeGraphMapping(
      std::shared_ptr<Graph> partial_eval_shape_graph,
      std::unordered_map<Value*, Value*> enclosing_graph_value_to_shape_graph_input,
      std::unordered_map<Value*, int64_t> graph_output_to_symbolic_shape_dim)
      : partial_eval_shape_graph(std::move(partial_eval_shape_graph)),
        enclosing_graph_value_to_shape_graph_input_(std::move(enclosing_graph_value_to_shape_graph_input)),
        graph_output_to_symbolic_shape_dim_(std::move(graph_output_to_symbolic_shape_dim)){};

  std::shared_ptr<Graph> partial_eval_shape_graph;  // 部分求值的形状图
  std::unordered_map<Value*, Value*> enclosing_graph_value_to_shape_graph_input_;  // 包含图中值到形状图输入的映射
  std::unordered_map<Value*, int64_t> graph_output_to_symbolic_shape_dim_;  // 图输出到符号形状维度的映射
};

// 在图上传播形状并构建大型形状计算图
TORCH_API std::optional<ShapeComputeGraphMapping>
PropagateShapesAndBuildLargeShapeComputeGraph(
    std::shared_ptr<Graph>& graph,
    Node* beg,
    Node* end);

// 在形状计算图中不插入完整张量形状，而是依赖部分求值流水线传播信息。
// 这是我们传播非完整形状信息能力的一个良好代理。
TORCH_API bool setSymbolicShapeAnalysisTestMode(bool value);

// 检查符号形状分析测试模式是否启用
TORCH_API bool symbolicShapeAnalysisTestModeEnabled();

// SSAInput定义为IValue或c10::SymbolicShape的可变类型
using SSAInput = std::variant<IValue, c10::SymbolicShape>;

// 计算操作的符号形状
TORCH_API std::optional<std::vector<c10::SymbolicShape>>
calculateSymbolicShapesOnOp(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& inputs);
} // namespace jit
} // namespace torch
```